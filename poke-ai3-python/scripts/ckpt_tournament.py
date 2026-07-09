#!/usr/bin/env python
"""checkpoint を恣意性なく多段選抜し、A/B 手法のレーティングを比較する orchestrator。

設計: docs/poke-ai3/20260620_2200_チェックポイント自動選抜トーナメントとABレーティング設計.md
      (本実装の合意は grill-me セッションで更新。下記 funnel 仕様を参照)

2 モード:
  funnel: 1 手法を連続学習しつつ多段選抜し、最終生存 checkpoint を集める。
    - epochs-per-step ごとに snapshot を作る (work ckpt を前進させ続ける。ロールバックなし)。
    - warmup: 開始から warmup-steps の間は単調増加期とみなし head-to-head せず素通り
      (snapshot も残さない)。
    - 0回戦: 新 snapshot を「リセット以降の全履歴」と対戦させ、新を負かした古 snapshot
      群の総当たり勝者を 1回戦突破 (ピーク) とする。放出後 history を空にして再開。
    - 1回戦突破が peaks-per-rr 個たまったら総当たりで勝率 1 位を 2回戦突破に。残り削除。
    - 2回戦突破が finalists-target 個たまったら終了。結果を JSON 出力。
    - ステップ境界で state JSON を保存し --resume で継続可能。
  rate: 複数手法の最終生存 (funnel の JSON) を集め、総当たり + Bradley-Terry で
    レーティング。手法ごとの平均レートを比較する。

head-to-head は全て policy-only (確率着手 / 先後入れ替え)。勝率 50% 超で勝ち
(引き分けは勝ち数比較から除外)。
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import secrets
import shutil
import subprocess
import sys
import re
from pathlib import Path

import numpy as np

from poke_ai3_train.bradley_terry import PairResult, bradley_terry_ratings

# 出力先はリポジトリルート相対 (data/poke-ai3)。このファイルは
# <repo>/poke-ai3-python/scripts/ckpt_tournament.py なので parents[2] がルート。
# 環境変数 POKE_AI3_DATA_DIR があればそれを優先 (別ディスクへ退避する場合用)。
import os

DATA = Path(os.environ.get("POKE_AI3_DATA_DIR",
                           Path(__file__).resolve().parents[2] / "data" / "poke-ai3"))
TDIR = DATA / "tournament"

_RESULT = re.compile(r"RESULT a_win=(\d+) b_win=(\d+) draw=(\d+)")


def _parse_value_target(s: str) -> bool:
    """--value-target の max/expected を bool (expected=True) へ。誤字は弾く。"""
    if s == "expected":
        return True
    if s == "max":
        return False
    raise argparse.ArgumentTypeError("value-target は max か expected")

# head-to-head は eval_ckpt_vs_ckpt を毎ペア subprocess 起動していたが、計測上 1 回 3s の
# うち ~2.8s が torch import + Agent ロード + CUDA Graph 構築の固定費で、512 試合の実計算は
# ~0.3s だった (experiments/poke-ai3 20260625)。全履歴比較 funnel では同じ checkpoint を
# 何百ペアも対戦させるため、この固定費が支配的だった。そこで eval をこのプロセス内に取り込み、
# Agent を checkpoint パスでキャッシュ (graph 込みでも 1 個 ~10MB と軽量) して使い回す。
# executor はペア毎に作り直す (40 連続生成でもリーク無しを確認済み) ことで pair 間の
# in-flight 試合混入を避ける。これで per-pair が ~2.8s → ~0.4s に短縮する。
_EVAL_SIMS = 64            # policy-only なので未使用だが executor 構築に必要 (旧既定と一致)
_EVAL_SIM_CONCURRENCY = 1
_EVAL_SEARCH_MIN = 4
_EVAL_SEARCH_MAX = 8

_AGENT_CACHE: dict[Path, "Agent"] = {}
_EXEC_FACTORY = None
# 学習ブロックをまたいで使い回す常駐セッション (初回 train_block_to で遅延構築)。
_TRAIN_SESSION = None

# eval (head-to-head) 用 battle_seed 生成器。以前は head_to_head が Rust 既定 (battle_seed=1)
# に静かに落ち、別 rate 呼び出しでも全ペアが seed=1 固定 → セル間ノイズが人為相関し
# 「再現性が高い」と誤認する評価バグがあった (experiments 20260707_2053 参照)。対策として
# eval seed は「既定ランダム (衝突しない) + 使った base を必ずログ + 再現時のみ --eval-seed で
# 明示ピン」とし、head_to_head は battle_seed を必須引数にして暗黙既定を根絶する。
_EVAL_RNG: "random.Random | None" = None


def init_eval_seed(base: int | None) -> int:
    """eval 用 battle_seed 生成器を初期化し、使った base をログして返す。base 省略時は
    ランダム。draw_eval_seed() が base から per-call の独立 seed を払い出す (ペアごとに
    独立。base を固定すれば総当たり全体を再現できる)。"""
    global _EVAL_RNG
    if base is None:
        base = secrets.randbits(63)
    _EVAL_RNG = random.Random(base)
    print(f"[eval] battle_seed base = {base}  (--eval-seed で固定可 / 省略時ランダム・毎回独立)",
          flush=True)
    return base


def draw_eval_seed() -> int:
    """head-to-head 1 回分の battle_seed を払い出す。init_eval_seed 未呼び出しでも
    取りこぼさないようその場でランダム初期化する。head_to_head は base と base+1 を
    先後 2 サイドに使うので、払い出しは 62bit に抑えて衝突余地を残す。"""
    global _EVAL_RNG
    if _EVAL_RNG is None:
        init_eval_seed(None)
    return _EVAL_RNG.getrandbits(62)


# ---------------------------------------------------------------- head-to-head


def _eval_agent(path: Path, num_games: int) -> "Agent":
    """checkpoint パスで Agent をキャッシュ。CUDA Graph 上限は num_games に揃える。"""
    from poke_ai3_train.agent import Agent

    agent = _AGENT_CACHE.get(path)
    if agent is None:
        agent = Agent(device=None, checkpoint_path=path, infer_max_batch_size=num_games)
        _AGENT_CACHE[path] = agent
    return agent


def head_to_head(
    new: Path, old: Path, n_per_side: int, stage: str, num_games: int,
    randomize: bool = False, crit_enabled: bool = False, *, battle_seed: int,
) -> tuple[int, int, int]:
    """policy-only で new vs old を先後入れ替えて対戦。new 視点 (win, loss, draw)。

    in-process 評価: Agent をキャッシュし、executor をペア毎に生成して collect_results で
    集計する (旧実装は eval_ckpt_vs_ckpt を毎ペア subprocess 起動していた)。

    battle_seed は必須 (キーワード専用)。Rust 既定 (=1) への暗黙フォールバックを禁じ、
    呼び出し側が draw_eval_seed() 等で明示的に seed を供給する。先後2サイドは
    battle_seed と battle_seed+1 を使う。"""
    global _EXEC_FACTORY
    from poke_ai3_train.train_loop import get_rust_async_executor_wrapper
    from poke_ai3_train.eval_ckpt_vs_ckpt import collect_results

    if _EXEC_FACTORY is None:
        _EXEC_FACTORY = get_rust_async_executor_wrapper()
    a_new = _eval_agent(new, num_games)
    a_old = _eval_agent(old, num_games)
    win = loss = draw = 0
    for k, new_is_p1 in enumerate((True, False)):
        agent_a, agent_b = (a_new, a_old) if new_is_p1 else (a_old, a_new)
        executor = _EXEC_FACTORY(
            num_games, num_games, None, "local", randomize, crit_enabled, stage,
            _EVAL_SIMS, _EVAL_SIM_CONCURRENCY, _EVAL_SEARCH_MIN, _EVAL_SEARCH_MAX,
            False, False, battle_seed=battle_seed + k, policy_only=True,
        )
        a_win, b_win, d = collect_results(
            executor, agent_a, agent_b, n_per_side, 0.0,
            print_every=0, quiet_progress=True,
        )
        del executor
        win += a_win if new_is_p1 else b_win
        loss += b_win if new_is_p1 else a_win
        draw += d
    return win, loss, draw


def winrate_vs(new: Path, old: Path, args) -> float:
    """new の old に対する勝率 (引き分け除外)。"""
    w, l, d = head_to_head(new, old, args.n_per_side, args.stage, args.num_games,
                           args.randomize, args.crit_enabled,
                           battle_seed=draw_eval_seed())
    decided = w + l
    wr = w / decided if decided else 0.5
    mark = "○新勝ち" if wr > 0.5 else "●旧勝ち"
    print(f"  [{mark}] {new.stem} vs {old.stem}: 新勝率={wr:.3f} (W={w} L={l} D={d})",
          flush=True)
    return wr


# ---------------------------------------------------------------- round robin


def round_robin_winner(pool: list[Path], args) -> Path:
    """pool 総当たり (先後入れ替え) で総勝率 1 位を返す。"""
    score = {p: 0 for p in pool}
    decided = {p: 0 for p in pool}
    for a, b in itertools.combinations(pool, 2):
        w, l, d = head_to_head(a, b, args.n_per_side, args.stage, args.num_games,
                               args.randomize, args.crit_enabled,
                               battle_seed=draw_eval_seed())
        score[a] += w
        score[b] += l
        decided[a] += w + l
        decided[b] += w + l
    rates = {p: (score[p] / decided[p] if decided[p] else 0.0) for p in pool}
    ranked = sorted(pool, key=lambda p: rates[p], reverse=True)
    print("  総当たり結果:")
    for p in ranked:
        print(f"    {p.stem}: 総勝率={rates[p]:.3f} (W={score[p]} / {decided[p]})")
    return ranked[0]


# ---------------------------------------------------------------- peak detector


class AllHistoryPeakDetector:
    """新 snapshot をリセット以降の「全履歴」と対戦させ、新を負かした古 snapshot 群の
    総当たり勝者を 1回戦突破 (ピーク) として放出する。

    学習初期の単調増加期は新が全てに勝つのでピークは出ない (warmup で素通りさせる前提)。
    戦略が一巡すると遠い過去が新に勝つようになり、その瞬間を循環ピークとみなす。近接バイアス
    (直近の自分には勝てる) は「新が勝つ → トリガーしない」ので全比較でも害が出にくい。
    放出後は history を完全に空にして (方式A) 現在の最新から再蓄積する。
    """

    def __init__(self, winrate, pick_winner, discard, name: str = "0回戦") -> None:
        self.winrate = winrate  # winrate(new, old) -> float (new の勝率)
        self.pick_winner = pick_winner  # pick_winner(pool) -> Path (総当たり勝者)
        self.discard = discard
        self.name = name
        self.history: list[Path] = []

    def feed(self, item: Path) -> Path | None:
        # 最新(item)が負ける (item の勝率 < 0.5) 古い snapshot を集める。
        beaten_by = [old for old in self.history if self.winrate(item, old) < 0.5]
        if not beaten_by:
            # 最新が全履歴に勝ち越し: まだ上り坂。history に追加。
            self.history.append(item)
            return None
        if len(beaten_by) == 1:
            peak = beaten_by[0]
        else:
            print(f"  ({self.name}) 最新を倒した古 snapshot {len(beaten_by)} 個 → 総当たりで絞り込み")
            peak = self.pick_winner(beaten_by)
        print(f"  ({self.name}) ピーク検出: {peak.stem} 昇格 "
              f"(最新 {item.stem} が過去に敗北)。history リセット。", flush=True)
        # peak 以外の history と、負けた最新 item を全て削除。
        for h in self.history:
            if h is not peak:
                self.discard(h)
        self.discard(item)
        self.history = []
        return peak


# ---------------------------------------------------------------- training


def train_block_to(work: Path, block_target: int, args) -> None:
    """work を block_target epoch まで学習する (常駐 TrainSession を使い回す)。
    epochs_per_step ごとに <work_stem>_ep<epoch>.pt を中間保存し、funnel 側はブロック
    終了後にそれらを epoch 順に消費する。work が既に block_target 到達済みなら no-op。

    旧実装はブロックごとに `uv run train-loop` をサブプロセス起動していたが、その固定費
    (uv 再ビルド判定 + torch import + Agent ロード + CUDA Graph 構築) を排すため、
    初回呼び出しで TrainSession を 1 回だけ構築し、以降のブロックはそれを run_to で前進
    させ続ける (executor / Agent はプロセス常駐)。"""
    _ensure_train_session(work, args).run_to(
        block_target, snapshot_every=args.epochs_per_step, checkpoint_path=work,
    )


def _ensure_train_session(work: Path, args):
    """常駐 TrainSession を遅延構築して返す (初回のみ構築、以降は使い回す)。"""
    global _TRAIN_SESSION
    if _TRAIN_SESSION is None:
        from poke_ai3_train.train_loop import TrainSession

        # スループット系の未指定 (None) は train-loop の既定に委ねる。max_batch_size は
        # None を渡すと TrainSession 側で num_games*sim_concurrency*2*3/7 を自動算出
        # (train_num_games 32 * sim_concurrency 16 なら 439)。trajectories_threshold=128 /
        # minibatch_size=256 は従来どおり。それ以外 (device/backend/nash/lr/battle_seed/
        # model_config 等) も train-loop 既定に一致させ、funnel の学習挙動を不変に保つ。
        _TRAIN_SESSION = TrainSession(
            num_games=args.train_num_games,
            max_batch_size=args.train_max_batch_size,
            trajectories_threshold=(args.train_trajectories_threshold
                                    if args.train_trajectories_threshold is not None else 128),
            sleep_seconds=0.0,
            device=None,
            checkpoint_path=work,
            backend="local",
            randomize=args.randomize,
            crit_enabled=args.crit_enabled,
            stage=args.stage,
            sims=args.sims,
            sim_concurrency=args.sim_concurrency,
            search_turn_min=args.search_turn_min,
            search_turn_max=args.search_turn_max,
            depth_skew=args.depth_skew,
            battle_seed=getattr(args, "train_battle_seed", None),
            minibatch_size=(args.train_minibatch_size
                            if args.train_minibatch_size is not None else 256),
            supervised_epochs=args.train_supervised_epochs,
            nash_learning_rate=getattr(args, "nash_learning_rate", 1.5),
            value_target_expected=getattr(args, "value_target_expected", False),
        )
    return _TRAIN_SESSION


def configure_enemies(work: Path, enemies: list[Path], args) -> None:
    """敵混合学習: 直近K個の敵を game_id ルーティングに割り当て、常駐セッションに
    役割テーブルと infer_fn を設定する。しきい値方式: 自己対戦=round(n*r) game を
    先頭 gid に割り当て (role 0)、残りを敵K個へ round-robin で均等分割 (role 1)。
    r=--self-play-ratio。K=1,r=0.5 は従来の 50/50 と一致。E=0 (敵なし/warmup) では
    全自己対戦に戻す。"""
    from poke_ai3_train.eval_ckpt_vs_ckpt import infer_step_pool

    sess = _ensure_train_session(work, args)
    n = sess.num_games
    e = len(enemies)
    roles = np.zeros(n, dtype=np.int64)
    enemy_by_game = {}
    enemy_labels: dict[int, str] = {}
    if e > 0:
        # 敵 Agent は _AGENT_CACHE を流用 (gate/finalist と同一インスタンス = peak 二役)。
        # CUDA Graph 上限は eval と揃え (args.num_games)、超過行は eager フォールバックで安全。
        agents = [_eval_agent(p, args.num_games) for p in enemies]
        r = getattr(args, "self_play_ratio", 0.5)
        s = round(n * r)  # 自己対戦 game 数 (先頭 gid に割当)
        # role: 1=敵 policy-only (従来) / 2=敵も先読み (--enemy-lookahead)。
        enemy_role = 2 if getattr(args, "enemy_lookahead", False) else 1
        for gid in range(s, n):
            idx = (gid - s) % e
            roles[gid] = enemy_role
            enemy_by_game[gid] = agents[idx]
            enemy_labels[gid] = enemies[idx].stem
    sess.enemy_labels = enemy_labels
    sess.executor.set_roles(roles)
    if enemy_by_game:
        agent = sess.agent
        sess.infer_fn = lambda ex: infer_step_pool(ex, agent, enemy_by_game)
        print(f"  [enemy] {e} 敵混合 (r={getattr(args, 'self_play_ratio', 0.5)}): "
              f"自己対戦={int((roles == 0).sum())}/{n} game, "
              f"敵={[p.stem for p in enemies]}", flush=True)
    else:
        sess.infer_fn = None


_EP_RE = re.compile(r"_ep(\d+)\.pt$")


def list_snapshots(work: Path, lo: int, hi: int) -> list[tuple[int, Path]]:
    """work と同 dir の <work_stem>_ep<epoch>.pt のうち lo < epoch <= hi のものを
    epoch 昇順で返す。"""
    out: list[tuple[int, Path]] = []
    for p in work.parent.glob(f"{work.stem}_ep*.pt"):
        m = _EP_RE.search(p.name)
        if m is None:
            continue
        e = int(m.group(1))
        if lo < e <= hi:
            out.append((e, p))
    out.sort(key=lambda t: t[0])
    return out


# ---------------------------------------------------------------- funnel mode


def funnel(args) -> None:
    init_eval_seed(getattr(args, "eval_seed", None))
    TDIR.mkdir(parents=True, exist_ok=True)
    # 作業用 ckpt は <tag>.pt。train-loop の --snapshot-every が <tag>_ep<epoch>.pt を
    # 自前で生成し、それをそのまま funnel の snapshot として使う (copy/rename 不要)。
    work = TDIR / f"{args.tag}.pt"
    state_path = TDIR / f"{args.tag}_state.json"

    # --train-block-epochs バリデーション (未指定なら epochs_per_step = 従来挙動)。
    block_epochs = args.train_block_epochs or args.epochs_per_step
    if block_epochs < args.epochs_per_step:
        raise SystemExit(
            f"--train-block-epochs ({block_epochs}) は "
            f"--epochs-per-step ({args.epochs_per_step}) 以上である必要があります。")
    if block_epochs % args.epochs_per_step != 0:
        raise SystemExit(
            f"--train-block-epochs ({block_epochs}) は "
            f"--epochs-per-step ({args.epochs_per_step}) の倍数である必要があります。")

    def discard(item) -> None:
        p = Path(item)
        # 削除する checkpoint の eval Agent をキャッシュから外し VRAM を解放する
        # (_AGENT_CACHE は evict しないため、放置すると削除済み ckpt の Agent が
        # GPU に居座り単調増加する。discard 連動で pop して bound する)。
        _AGENT_CACHE.pop(p, None)
        # work dir 内の自前 snapshot のみ削除 (seed checkpoint は触らない)。
        if p.parent == TDIR and p.exists():
            p.unlink()

    wr = lambda new, old: winrate_vs(new, old, args)
    det = AllHistoryPeakDetector(wr, lambda pool: round_robin_winner(pool, args), discard)
    peaks: list[Path] = []
    finalists: list[Path] = []
    # 敵混合学習 (--enemy-window K>=1): gate で検出した peak を貯める append-only の敵列。
    # finalist 選抜の peaks (2段構造・リセットあり) とは別管理で、学習には末尾K個を混ぜる。
    # #4/#5 の削除を撤廃したので敵列のファイルは run 終了まで健在 = resume で再ロード可能。
    enemy_pool: list[Path] = []
    enemy_mode = getattr(args, "enemy_window", 0) >= 1

    # ---- resume か新規開始か ----
    if args.resume and state_path.exists():
        st = json.loads(state_path.read_text())
        base = st["base"]
        epoch = st["epoch"]
        warmup_until = st["warmup_until"]
        # 旧 state には block_target が無い → epoch で補完 (次ブロックから新方式)。
        block_target = st.get("block_target", epoch)
        det.history = [Path(p) for p in st["history"]]
        peaks = [Path(p) for p in st["peaks"]]
        finalists = [Path(p) for p in st["finalists"]]
        enemy_pool = [Path(p) for p in st.get("enemy_pool", [])]
        print(f"[funnel] resume: ep{epoch} block_target={block_target} "
              f"history={len(det.history)} peaks={len(peaks)} "
              f"finalists={len(finalists)} enemy_pool={len(enemy_pool)}", flush=True)
    else:
        if args.start is not None:
            shutil.copy(args.start, work)
            import torch
            base = int(torch.load(work, map_location="cpu").get("training_step", 0))
            start_name = args.start.name
        else:
            if work.exists():
                work.unlink()
            base = 0
            start_name = "<random-init>"
        epoch = base
        block_target = base
        warmup_until = base + args.warmup_steps * args.epochs_per_step
        print(f"[funnel] method={args.tag} start={start_name} base_step={base} "
              f"warmup_steps={args.warmup_steps} (~ep{warmup_until}) "
              f"peaks/rr={args.peaks_per_rr} finalists_target={args.finalists_target} "
              f"epochs/step={args.epochs_per_step} block_epochs={block_epochs} "
              f"depth_skew={args.depth_skew} "
              f"value_target={'expected' if getattr(args, 'value_target_expected', False) else 'max'}",
              flush=True)

    def save_state() -> None:
        state_path.write_text(json.dumps({
            "base": base, "epoch": epoch, "warmup_until": warmup_until,
            "block_target": block_target,
            "history": [str(p) for p in det.history],
            "peaks": [str(p) for p in peaks],
            "finalists": [str(p) for p in finalists],
            "enemy_pool": [str(p) for p in enemy_pool],
        }, ensure_ascii=False, indent=2))

    peaks_emitted = len(finalists) * args.peaks_per_rr + len(peaks)

    def consume(snap_epoch: int, snap: Path) -> None:
        """snapshot を 1 個処理する (warmup なら削除、それ以外は detector へ feed)。"""
        nonlocal epoch, peaks, peaks_emitted
        epoch = snap_epoch
        if snap_epoch <= warmup_until:
            print(f"  (warmup) ep{snap_epoch} は単調増加期とみなし head-to-head 省略。",
                  flush=True)
            discard(snap)
            save_state()
            return
        peak = det.feed(snap)
        save_state()
        if peak is None:
            return
        peaks_emitted += 1
        peaks.append(peak)
        # 敵混合学習: 検出 peak を append-only の敵列にも積む (学習には末尾K個を混ぜる)。
        if enemy_mode:
            enemy_pool.append(peak)
        print(f"  >> 1回戦突破 {len(peaks)}/{args.peaks_per_rr}: {peak.stem}", flush=True)
        if len(peaks) < args.peaks_per_rr:
            return
        print(f"\n====== 2回戦 総当たり ({len(peaks)} 者) ======", flush=True)
        finalist = round_robin_winner(peaks, args)
        print(f"  >> 2回戦突破: {finalist.stem}", flush=True)
        # #4 撤廃: 非勝者 peak を削除しない (敵列や後の参照のため run 終了まで残す)。
        # ディスクの主消費は detector の非peak history 掃除 (#1) で従来どおり回収される。
        peaks = []
        finalists.append(finalist)
        print(f"  >>>> 最終生存 {len(finalists)}/{args.finalists_target}: "
              f"{finalist.stem}", flush=True)
        save_state()

    def prep_block() -> None:
        """敵混合モードなら、このブロックの敵 (敵列の末尾K) を役割/推論に設定する。
        K=0 (通常 funnel) では何もしない (常駐セッションは train_block_to が構築)。"""
        if enemy_mode:
            configure_enemies(work, enemy_pool[-args.enemy_window:], args)

    # resume 時、中断したブロックを block_target まで完遂してから消費する
    # (train-loop は work から自動継続。到達済みなら即終了の no-op)。
    if args.resume and state_path.exists() and block_target > epoch:
        print(f"\n###### resume: train block -> ep{block_target} "
              f"(snapshot/{args.epochs_per_step}) ######", flush=True)
        prep_block()
        train_block_to(work, block_target, args)

    # 前ブロックで生成済みだが未処理の snapshot を先に消費する。
    for e, snap in list_snapshots(work, epoch, block_target):
        consume(e, snap)
        if len(finalists) >= args.finalists_target:
            break

    while len(finalists) < args.finalists_target:
        if epoch - base >= args.max_added_epochs:
            print(f"\n!! 異常: max_added_epochs({args.max_added_epochs}) 到達。"
                  f"finalists={len(finalists)}/{args.finalists_target} "
                  f"(1回戦突破 累計={peaks_emitted}, 収集中 peaks={len(peaks)}, "
                  f"history={len(det.history)})。一巡せず単調増加が続いた可能性。", flush=True)
            break
        prev_target = block_target
        block_target = min(block_target + block_epochs, base + args.max_added_epochs)
        if block_target <= prev_target:
            break
        print(f"\n###### train block -> ep{block_target} "
              f"(snapshot/{args.epochs_per_step}) ######", flush=True)
        prep_block()
        train_block_to(work, block_target, args)
        # ブロックで生成された snapshot を epoch 昇順に消費する。
        for e, snap in list_snapshots(work, epoch, block_target):
            consume(e, snap)
            if len(finalists) >= args.finalists_target:
                break
        # #5 撤廃: finalists_target 到達後の余剰 snapshot を削除しない。神経質な全削除を
        # やめ、run 終了まで残す (ディスク主消費は #1 の非peak history 掃除で回収済み)。

    out = {
        "method": args.tag,
        "start": str(args.start) if args.start else "<random-init>",
        "depth_skew": args.depth_skew,
        "value_target": "expected" if getattr(args, "value_target_expected", False) else "max",
        "search_turn": [args.search_turn_min, args.search_turn_max],
        "peaks_emitted": peaks_emitted,
        "finalists": [str(p) for p in finalists],
    }
    out_path = TDIR / f"{args.tag}_finalists.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[funnel] 完了。最終生存 {len(finalists)} 個 → {out_path}", flush=True)
    for p in finalists:
        print(f"  {p}")


# ---------------------------------------------------------------- rate mode


def rate(args) -> None:
    init_eval_seed(getattr(args, "eval_seed", None))
    # 各 funnel JSON を読み、(label, method, path) のプールを作る。
    entries: list[tuple[str, str, Path]] = []
    for jp in args.funnel_json:
        d = json.loads(Path(jp).read_text())
        method = d["method"]
        # finalists (新) / secondaries (旧 JSON) 両対応。
        members = d.get("finalists", d.get("secondaries", []))
        for i, p in enumerate(members):
            entries.append((f"{method}#{i}", method, Path(p)))
    if len(entries) < 2:
        raise SystemExit("レート戦には 2 個以上の checkpoint が必要")
    labels = [e[0] for e in entries]
    path_of = {e[0]: e[2] for e in entries}
    method_of = {e[0]: e[1] for e in entries}
    print(f"[rate] プール {len(entries)} 個: {labels}", flush=True)

    pairs: list[PairResult] = []
    for (la, _, pa), (lb, _, pb) in itertools.combinations(entries, 2):
        w, l, d = head_to_head(pa, pb, args.n_per_side, args.stage, args.num_games,
                               args.randomize, args.crit_enabled,
                               battle_seed=draw_eval_seed())
        print(f"  {la} vs {lb}: {la}勝={w} {lb}勝={l} 引分={d}", flush=True)
        pairs.append(PairResult(la, lb, w, l, d))

    ratings = bradley_terry_ratings(labels, pairs)
    print("\n=== Bradley-Terry レーティング (平均 0) ===")
    for la in sorted(labels, key=lambda x: ratings[x], reverse=True):
        print(f"  {la} ({method_of[la]}): {ratings[la]:+.1f}  [{path_of[la].name}]")

    by_method: dict[str, list[float]] = {}
    for la in labels:
        by_method.setdefault(method_of[la], []).append(ratings[la])
    print("\n=== 手法ごと平均レート ===")
    ranked = sorted(by_method, key=lambda m: sum(by_method[m]) / len(by_method[m]),
                    reverse=True)
    for m in ranked:
        vals = by_method[m]
        print(f"  {m}: 平均={sum(vals) / len(vals):+.1f} (n={len(vals)}, {[f'{v:+.1f}' for v in vals]})")
    print(f"\n==> 優れた手法: {ranked[0]}", flush=True)


# ---------------------------------------------------------------- exploit mode


def exploit(args) -> None:
    """固定 target への best-response (exploiter) を学習し exploitability を測る。

    exploiter は shared_init 等から開始し、敵を target 1 体に固定 (self_play_ratio=0 で
    全 game が exploiter vs target)。funnel と違い検出 peak を敵列へ積まない純 best-response。
    --eval-every ごとに 1 ブロック学習し、その都度 exploiter vs target を policy-only で測る。
    勝率が更新されるたびピーク重みを <tag>_peak.pt に退避し、最終的な exploiter はこの
    ピーク重みとする (最終 epoch ではなくカーブ最大の重みを採用)。--patience>0 なら勝率が
    patience 回連続で更新されなくなった時点で early-stop する (ピークアウト検出)。
    exploitability 推定値 = 学習カーブ上の exploiter 勝率の最大値 (低い target ほど
    unexploitable)。--resume で予算を延長できる。"""
    init_eval_seed(getattr(args, "eval_seed", None))
    TDIR.mkdir(parents=True, exist_ok=True)
    work = TDIR / f"{args.tag}.pt"
    peak_path = TDIR / f"{args.tag}_peak.pt"   # カーブ最大勝率時点の重みを退避
    state_path = TDIR / f"{args.tag}_state.json"
    target = args.target
    if not target.exists():
        raise SystemExit(f"target が見つかりません: {target}")

    # 敵は target 1 体固定・全 game を exploiter vs target に (Q4=A: target は policy-only)。
    args.self_play_ratio = 0.0
    args.enemy_lookahead = False
    # train_block_to は epochs_per_step を snapshot 間隔に使う。eval-every を 1 ブロックに。
    args.epochs_per_step = args.eval_every
    patience = getattr(args, "patience", 0)  # 0=early-stop 無効 (固定予算)

    curve: list[list[float]] = []  # [[epoch, winrate], ...]
    best: list[float] | None = None
    no_improve = 0                 # 連続で best を更新できなかった eval 数

    if args.resume and state_path.exists():
        st = json.loads(state_path.read_text())
        base = st["base"]
        epoch = st["epoch"]
        curve = st["curve"]
        best = st.get("best")
        no_improve = st.get("no_improve", 0)
        print(f"[exploit] resume: ep{epoch} base={base} evals={len(curve)} "
              f"best={best} no_improve={no_improve}", flush=True)
    else:
        if args.start is not None:
            shutil.copy(args.start, work)
            import torch
            base = int(torch.load(work, map_location="cpu").get("training_step", 0))
            start_name = args.start.name
        else:
            if work.exists():
                work.unlink()
            base = 0
            start_name = "<random-init>"
        epoch = base
        # 古い自前 snapshot / ピーク退避を掃除。
        for p in TDIR.glob(f"{args.tag}_ep*.pt"):
            p.unlink()
        if peak_path.exists():
            peak_path.unlink()
        print(f"[exploit] tag={args.tag} start={start_name} target={target.name} "
              f"base={base} eval_every={args.eval_every} "
              f"max_added_epochs={args.max_added_epochs} patience={patience} "
              f"value_target={'expected' if getattr(args, 'value_target_expected', False) else 'max'}",
              flush=True)

    def save_state() -> None:
        state_path.write_text(json.dumps({
            "base": base, "epoch": epoch, "target": str(target),
            "curve": curve, "best": best, "no_improve": no_improve,
        }, ensure_ascii=False, indent=2))

    def eval_now() -> float:
        # work は再学習で in-place 更新されるため、eval 前に stale な Agent を必ず捨てて
        # ディスク上の最新重みを再ロードさせる (_eval_agent はパス永続キャッシュで mtime を
        # 見ないため、これをしないと最初の eval 時点の重みを測り続けるバグになる)。
        _AGENT_CACHE.pop(work, None)
        w, l, d = head_to_head(work, target, args.n_per_side, args.stage,
                               args.num_games, args.randomize, args.crit_enabled,
                               battle_seed=draw_eval_seed())
        decided = w + l
        wr = w / decided if decided else 0.5
        print(f"  [exploit] ep{epoch} exploiter vs {target.stem}: "
              f"勝率={wr:.3f} (W={w} L={l} D={d})", flush=True)
        return wr

    while epoch - base < args.max_added_epochs:
        block_target = min(epoch + args.eval_every, base + args.max_added_epochs)
        if block_target <= epoch:
            break
        print(f"\n###### exploit train block -> ep{block_target} ######", flush=True)
        configure_enemies(work, [target], args)  # self_play_ratio=0 → 全 game vs target
        train_block_to(work, block_target, args)
        epoch = block_target
        # ブロック終了時の work (=最新 exploiter) を直接測る。中間 snapshot は不要なので掃除。
        for p in TDIR.glob(f"{args.tag}_ep*.pt"):
            p.unlink()
        wr = eval_now()
        curve.append([epoch, wr])
        if best is None or wr > best[1]:
            best = [epoch, wr]
            no_improve = 0
            shutil.copy(work, peak_path)   # ピーク重みを退避 (最終採用はこれ)
        else:
            no_improve += 1
        print(f"  [exploit] best so far: 勝率={best[1]:.3f} @ep{int(best[0])} "
              f"(no_improve={no_improve})", flush=True)
        save_state()
        if patience and no_improve >= patience:
            print(f"  [exploit] ピークアウト検出 (no_improve={no_improve} >= "
                  f"patience={patience}) → early-stop。ピーク @ep{int(best[0])} を採用。",
                  flush=True)
            break

    # 最終的な exploiter はカーブ最大勝率時点 (ピーク) の重み。work をピークで上書きし、
    # 下流 (psro のプール) が最終 epoch ではなくピーク重みを使うようにする。
    if peak_path.exists():
        shutil.copy(peak_path, work)
        _AGENT_CACHE.pop(work, None)

    out = {
        "tag": args.tag,
        "target": str(target),
        "start": str(args.start) if args.start else "<random-init>",
        "value_target": "expected" if getattr(args, "value_target_expected", False) else "max",
        "eval_every": args.eval_every,
        "patience": patience,
        "curve": curve,
        "exploitability": best[1] if best else None,
        "best_epoch": int(best[0]) if best else None,
    }
    out_path = TDIR / f"{args.tag}_exploit.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[exploit] 完了。exploitability (best 勝率) = "
          f"{out['exploitability']} @ep{out['best_epoch']} (採用=ピーク重み) → {out_path}",
          flush=True)


# ---------------------------------------------------------------- psro mode


def _fwd_shared_args(args) -> list[str]:
    """common + 学習側設定を子プロセス (exploit) へ引き継ぐ argv を作る。
    exploiter も中心と同じ探索/学習設定・stage・random/crit で回すため。"""
    argv = [
        "--n-per-side", str(args.n_per_side),
        "--num-games", str(args.num_games),
        "--stage", args.stage,
        "--random" if args.randomize else "--no-random",
        "--crit" if args.crit_enabled else "--no-crit",
        "--depth-skew", str(args.depth_skew),
        "--search-turn-min", str(args.search_turn_min),
        "--search-turn-max", str(args.search_turn_max),
        "--sims", str(args.sims),
        "--sim-concurrency", str(args.sim_concurrency),
        "--train-num-games", str(args.train_num_games),
        "--nash-learning-rate", str(args.nash_learning_rate),
        "--value-target",
        "expected" if getattr(args, "value_target_expected", False) else "max",
    ]
    for name, val in (
        ("--train-max-batch-size", args.train_max_batch_size),
        ("--train-trajectories-threshold", args.train_trajectories_threshold),
        ("--train-minibatch-size", args.train_minibatch_size),
        ("--train-supervised-epochs", args.train_supervised_epochs),
    ):
        if val is not None:
            argv += [name, str(val)]
    return argv


def psro(args) -> None:
    """PSRO ループ: 中心学習者 1 本を連続育成しつつ、毎イテレーションその時点の中心を
    凍結 target にした専用 best-response (exploiter) を作り、敵プールへ積む。

    中心学習者は常駐 TrainSession でプロセス内に生かし続け (optimizer/graph を温存)、
    相手は「自己対戦 self_play_ratio ＋ 最新 pool_size 個の exploiter 一様」。funnel の
    auto-peak 積みは使わず、敵は自作 exploiter プールに固定する。exploiter は別プロセスで
    exploit サブコマンドを起動し (中心とは別ネットなので global session の衝突を避ける)、
    その対 target 勝率 (exploitability) を毎 iter 記録する。iter を追って 50% へ下がれば
    穴が塞がった (ナッシュへ寄った) 証拠。--resume で iter を延長できる。"""
    init_eval_seed(getattr(args, "eval_seed", None))
    TDIR.mkdir(parents=True, exist_ok=True)
    ctag = args.tag
    work = TDIR / f"{ctag}.pt"
    state_path = TDIR / f"{ctag}_psro_state.json"
    shared_init = args.shared_init
    if not shared_init.exists():
        raise SystemExit(f"shared-init が見つかりません: {shared_init}")

    # 中心の敵設定: 自己対戦 r (--self-play-ratio) ＋ 敵 policy-only (auto-peak なし)。
    args.enemy_lookahead = False
    # train_block_to の snapshot 間隔 = 中心ブロック長 (末尾で 1 回だけ snapshot)。
    args.epochs_per_step = args.central_epochs

    pool: list[Path] = []           # 積んだ exploiter checkpoint (append-only)
    curve: list[list[float]] = []   # [[iter, central_epoch, exploitability], ...]

    if args.resume and state_path.exists():
        st = json.loads(state_path.read_text())
        base = st["base"]
        epoch = st["epoch"]
        it = st["iter"]
        pool = [Path(p) for p in st["pool"]]
        curve = st["curve"]
        print(f"[psro] resume: iter={it} ep{epoch} pool={len(pool)} "
              f"evals={len(curve)}", flush=True)
    else:
        shutil.copy(shared_init, work)
        import torch
        base = int(torch.load(work, map_location="cpu").get("training_step", 0))
        epoch = base
        it = 0
        # 古い中心 snapshot を掃除。
        for p in TDIR.glob(f"{ctag}_ep*.pt"):
            p.unlink()
        print(f"[psro] tag={ctag} shared_init={shared_init.name} base={base} "
              f"warmup_epochs={args.warmup_epochs} "
              f"central_epochs={args.central_epochs} exploiter_epochs={args.exploiter_epochs} "
              f"pool_size={args.pool_size} self_play_ratio={args.self_play_ratio} "
              f"max_iters={args.max_iters} "
              f"value_target={'expected' if getattr(args, 'value_target_expected', False) else 'max'}",
              flush=True)

    def save_state() -> None:
        state_path.write_text(json.dumps({
            "base": base, "epoch": epoch, "iter": it,
            "pool": [str(p) for p in pool], "curve": curve,
        }, ensure_ascii=False, indent=2))

    while it < args.max_iters:
        # ---- 1) 中心学習者を前進 (敵=最新 pool_size 個) ----
        # iter0 は warmup_epochs だけ育ててから最初の exploit (ep50 の未熟な中心を突くのは
        # 早すぎるため)。以降の iter は central_epochs ずつ。
        inc = args.warmup_epochs if it == 0 else args.central_epochs
        window = pool[-args.pool_size:]
        block_target = epoch + inc
        print(f"\n###### [psro] iter {it}: 中心学習 ep{epoch} -> ep{block_target} "
              f"({'warmup' if it == 0 else 'central'} {inc}ep, 敵 {len(window)} 体) ######",
              flush=True)
        configure_enemies(work, window, args)
        train_block_to(work, block_target, args)
        epoch = block_target
        for p in TDIR.glob(f"{ctag}_ep*.pt"):
            p.unlink()

        # ---- 2) 現時点の中心を凍結 target にし、別プロセスで exploiter を学習 ----
        target = TDIR / f"{ctag}_iter{it}_target.pt"
        shutil.copy(work, target)
        exp_tag = f"{ctag}_exp{it}"
        exp_json = TDIR / f"{exp_tag}_exploit.json"
        # exploiter の初期値: target 自身 (既定) から自分に勝つ方向へ微調整すると、始点が
        # 約 0.5 で「弱すぎる個体」にならず収束も速い。shared-init は 0.05 から長く登る旧方式。
        exp_start = target if args.exploiter_init == "target" else shared_init
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "exploit",
            "--target", str(target), "--tag", exp_tag,
            "--start", str(exp_start),
            "--eval-every", str(args.exploiter_eval_every),
            "--max-added-epochs", str(args.exploiter_epochs),
            "--patience", str(args.exploiter_patience),
            "--train-battle-seed", str(args.exploiter_battle_seed),
        ] + _fwd_shared_args(args)
        print(f"\n###### [psro] iter {it}: exploiter 学習 (target={target.name}) ######\n"
              f"  $ {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0 or not exp_json.exists():
            raise SystemExit(f"[psro] iter {it} exploiter 失敗 (exit={r.returncode})")
        exp = json.loads(exp_json.read_text())
        exploitability = exp["exploitability"]
        exp_ckpt = TDIR / f"{exp_tag}.pt"   # exploit が残すピーク重み exploiter ネット
        pool.append(exp_ckpt)
        curve.append([it, epoch, exploitability])
        print(f"\n[psro] iter {it} 完了: exploitability={exploitability} "
              f"(exploiter {exp_tag}, curve={exp['curve']})", flush=True)
        it += 1
        save_state()

    out = {
        "tag": ctag,
        "shared_init": str(shared_init),
        "warmup_epochs": args.warmup_epochs,
        "central_epochs": args.central_epochs,
        "exploiter_epochs": args.exploiter_epochs,
        "exploiter_patience": args.exploiter_patience,
        "exploiter_init": args.exploiter_init,
        "pool_size": args.pool_size,
        "self_play_ratio": args.self_play_ratio,
        "value_target": "expected" if getattr(args, "value_target_expected", False) else "max",
        "iters": it,
        "pool": [str(p) for p in pool],
        "curve": curve,   # [[iter, central_epoch, exploitability], ...]
        "final_central": str(work),
    }
    out_path = TDIR / f"{ctag}_psro.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[psro] 完了。{it} iter。exploitability 推移: "
          f"{[[c[0], c[2]] for c in curve]} → {out_path}", flush=True)


# ---------------------------------------------------------------- cli


def add_train_side_args(p: argparse.ArgumentParser) -> None:
    """funnel / exploit 共有の学習側設定 (A/B の比較対象・生成器の探索設定)。"""
    p.add_argument("--train-battle-seed", type=int, default=None,
                   help="学習 rollout の battle_seed。既定 None=毎 run ランダム独立。"
                        "対応比較 (両アームを同一 seed で回す) 用に固定できる。")
    p.add_argument("--depth-skew", type=float, default=1.0)
    p.add_argument("--search-turn-min", type=int, default=6)
    p.add_argument("--search-turn-max", type=int, default=12)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--sim-concurrency", type=int, default=16)
    p.add_argument("--train-num-games", type=int, default=32)
    p.add_argument("--train-max-batch-size", type=int, default=None,
                   help="学習 train-loop の --max-batch-size。未指定なら train-loop 既定。")
    p.add_argument("--train-trajectories-threshold", type=int, default=None,
                   help="学習 train-loop の --trajectories-threshold。未指定なら train-loop 既定。")
    p.add_argument("--train-minibatch-size", type=int, default=None,
                   help="学習 train-loop の --minibatch-size。未指定なら train-loop 既定(256)。")
    p.add_argument("--train-supervised-epochs", type=int, default=None,
                   help="学習 train-loop の --supervised-epochs (1バッチを何パスなめるか)。"
                        "未指定なら train-loop 既定(4)。")
    p.add_argument("--nash-learning-rate", type=float, default=1.5,
                   help="学習 train-loop の --nash-learning-rate (nash_weak 穏当化版の"
                        "更新率)。既定は train-loop と同じ 1.5。A/B 用に振れる。")
    p.add_argument("--value-target", dest="value_target_expected",
                   default=False, type=_parse_value_target,
                   help="value 教師の式。max (既定) は手ごと最大勝率、expected は均衡混合 "
                        "training_pi による期待勝率。A/B 用。")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-per-side", type=int, default=512,
                        help="1 対戦の片側試合数 (合計はこの 2 倍)。既定 512 → 1024 試合。")
    common.add_argument("--num-games", type=int, default=256,
                        help="eval の並列ゲーム数。policy-only head-to-head は学習が無く "
                             "並列度がそのまま batch を太らせGPU充填を上げるため大きめが有利。"
                             "Phase3 スイープで 64→256 が実バトル/s +33%% (384以降は頭打ち)。"
                             "max-batch-size は num-games に揃える (in-flight=num-games*2 で余裕)。")
    common.add_argument("--stage", type=str, choices=["3a", "3b", "3c"], default="3b")
    common.add_argument("--random", dest="randomize",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="学習(funnel)・評価(funnel選抜/rate)で16段ダメージロールを有効化。"
                             "既定 --no-random (決定論)。")
    common.add_argument("--crit", dest="crit_enabled",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="学習(funnel)・評価(funnel選抜/rate)で急所を有効化。"
                             "既定 --no-crit。")
    common.add_argument("--eval-seed", type=int, default=None,
                        help="eval(head-to-head)の battle_seed base。省略時はランダム"
                             "(毎回独立サンプル)で起動時にログ出力。再現したいときだけ整数を明示。"
                             "全ペアはこの base から独立 seed を払い出す(旧実装のように全ペア"
                             "seed=1 固定でノイズが人為相関する事故を防ぐ)。")

    f = sub.add_parser("funnel", parents=[common], help="1 手法を多段選抜")
    f.add_argument("--start", type=Path, default=None,
                   help="開始 checkpoint。省略時はランダム初期状態から学習開始。")
    f.add_argument("--tag", type=str, required=True, help="手法ラベル (例 A / B)")
    f.add_argument("--resume", action="store_true",
                   help="既存の <tag>_state.json から選抜の進行状態を復元して継続。")
    f.add_argument("--epochs-per-step", type=int, default=20,
                   help="snapshot 間隔(epoch)。")
    f.add_argument("--train-block-epochs", type=int, default=None,
                   help="1 ブロックで追加学習する最大 epoch 数 (この単位で snapshot を消費して "
                        "head-to-head 選抜する)。epochs-per-step の倍数であること。未指定なら "
                        "epochs-per-step (snapshot 毎に選抜)。学習は常駐 TrainSession が "
                        "プロセス内で継続するため、ブロック境界でのプロセス再起動は無い。")
    f.add_argument("--enemy-window", type=int, default=1,
                   help="敵混合学習の敵プール窓サイズ K。既定 1 (推奨)。gate 検出 peak の敵列 "
                        "(append-only) の末尾K個を学習敵として混ぜる。剩余バケット game_id%%(E+1) "
                        "で自己対戦と敵を均等割り (K=1→50%%, K=2→33%% ずつ)。敵側 P2 は "
                        "policy-only。0 で純自己対戦 funnel (従来挙動)。"
                        "K 掃引 (experiments 20260701_2246) で K=1 が baseline 比 +9.1 Elo で "
                        "最良、K>=2 は自己対戦比率が下がり優位消失のため既定を 1 に採用。")
    f.add_argument("--self-play-ratio", type=float, default=0.5,
                   help="敵混合時の自己対戦 game の割合 r (0<r<=1)。しきい値方式で "
                        "自己対戦=round(n*r) game、残りを敵K個へ均等分割。既定 0.5 "
                        "(自己対戦50%%/敵50%%、K=1 で従来の 50/50 と一致)。"
                        "r 掃引 (experiments 20260702_1312) では一時 r=0.4 が最良に見えたが、"
                        "r=0.4 の再現ラン (K1r04b) が +7.2→-3.6 と割れ run 運が支配的と判明。"
                        "比率の有意差は現データでは判定不能のため既定は素直な 0.5 に据える。"
                        "注意: K>=2 で未指定だと敵50%%をK分割する挙動になり、旧 enemy-window "
                        "掃引 (K2=33%%/K3=25%%) とは非互換。K>=2 掃引の再現には比率を明示せよ。")
    f.add_argument("--enemy-lookahead", action="store_true",
                   help="敵混合の敵 P2 も学習者 P1 と同じ探索設定 (search-turn/sims/depth-skew) "
                        "で着手させる (既定 off=policy-only)。role テーブル値 2 で表現。"
                        "生成コストは増える (敵ゲームも探索するため)。")
    f.add_argument("--warmup-steps", type=int, default=10,
                   help="開始から head-to-head を省略する step 数 (単調増加期の素通り)。")
    f.add_argument("--peaks-per-rr", type=int, default=3,
                   help="2回戦の総当たりに進める 1回戦突破数")
    f.add_argument("--finalists-target", type=int, default=3, help="集める最終生存数")
    f.add_argument("--max-added-epochs", type=int, default=4000,
                   help="開始からの追加 epoch 上限 (無限ループ防止)")
    # 学習側設定 (--train-battle-seed 含む。A/B の比較対象)。exploit でも同一設定を共有する。
    add_train_side_args(f)
    f.set_defaults(func=funnel)

    x = sub.add_parser(
        "exploit", parents=[common],
        help="固定 target への best-response (exploiter) を学習し exploitability を測る")
    x.add_argument("--target", type=Path, required=True,
                   help="突く対象の固定 checkpoint。exploiter はこれ 1 体のみを敵に学習する。")
    x.add_argument("--tag", type=str, required=True,
                   help="exploiter ラベル (例 EXP_VEXP_s1)。work/state/結果 JSON の接頭辞。")
    x.add_argument("--start", type=Path, default=None,
                   help="exploiter の開始 checkpoint (推奨: shared_init.pt)。"
                        "省略時はランダム初期状態から。")
    x.add_argument("--resume", action="store_true",
                   help="既存の <tag>_state.json から継続 (予算延長にも使う)。")
    x.add_argument("--eval-every", type=int, default=50,
                   help="exploiter vs target を測る間隔 (epoch)。この単位で 1 ブロック学習し、"
                        "各ブロック後に policy-only 勝率を記録する。既定 50。")
    x.add_argument("--max-added-epochs", type=int, default=200,
                   help="開始からの追加 epoch 上限。固定 target への best-response は収束が"
                        "速い想定で既定 200。まだ登っていれば --resume で延長する。"
                        "--patience>0 のときは early-stop の上限 (安全弁) として働く。")
    x.add_argument("--patience", type=int, default=0,
                   help="ピークアウト early-stop の忍耐。勝率が patience 回連続で best を "
                        "更新できなくなったら停止し、ピーク重み (<tag>_peak.pt) を採用。"
                        "既定 0=無効 (max-added-epochs までの固定予算)。ノイズ耐性のため 2 推奨。")
    add_train_side_args(x)
    x.set_defaults(func=exploit)

    p = sub.add_parser(
        "psro", parents=[common],
        help="PSRO ループ: 中心を育てつつ毎 iter exploiter を作り敵プールへ積む")
    p.add_argument("--tag", type=str, required=True,
                   help="中心学習者ラベル (work/state/結果 JSON の接頭辞)。")
    p.add_argument("--shared-init", type=Path, required=True,
                   help="中心・exploiter 両方の開始 checkpoint (shared_init.pt)。")
    p.add_argument("--resume", action="store_true",
                   help="既存の <tag>_psro_state.json から iter を継続 (延長にも使う)。")
    p.add_argument("--max-iters", type=int, default=6,
                   help="PSRO イテレーション数。既定 6 (パイロット)。--resume で延長。")
    p.add_argument("--warmup-epochs", type=int, default=200,
                   help="最初の exploit の前に中心を自己対戦で育てる epoch (iter0 の学習量)。"
                        "既定 200。ep50 程度の未熟な中心を突くのは早すぎるため。")
    p.add_argument("--central-epochs", type=int, default=50,
                   help="iter1 以降で 1 iter に中心学習者を前進させる epoch。既定 50。")
    p.add_argument("--exploiter-epochs", type=int, default=200,
                   help="各 iter の exploiter best-response の追加 epoch 上限 (early-stop の"
                        "安全弁)。既定 200。ピークアウトで早期停止するので大きめでよい。")
    p.add_argument("--exploiter-eval-every", type=int, default=25,
                   help="exploiter の eval 間隔 (epoch)。既定 25。この単位でピーク検出する。")
    p.add_argument("--exploiter-init", choices=["target", "shared-init"], default="target",
                   help="exploiter の初期値。target (既定)=その時点の中心自身から始めて自分に"
                        "勝つ方向へ微調整 (始点 ~0.5、弱すぎる個体にならず速い)。shared-init="
                        "共通初期値から (始点 ~0.05、長く登る旧方式)。")
    p.add_argument("--exploiter-patience", type=int, default=1,
                   help="exploiter のピークアウト early-stop 忍耐 (子 exploit へ --patience で"
                        "渡す)。勝率が patience 回連続で更新されなければ停止しピーク重みを採用。"
                        "既定 1 (更新なしで即停止)。ノイズが気になれば 2 以上。")
    p.add_argument("--pool-size", type=int, default=4,
                   help="中心の敵に混ぜる最新 exploiter 数 N (窓)。既定 4。")
    p.add_argument("--self-play-ratio", type=float, default=0.5,
                   help="中心学習の自己対戦 game 割合 r。既定 0.5 (残りを敵 N 個へ均等分割)。")
    p.add_argument("--exploiter-battle-seed", type=int, default=20260711,
                   help="exploiter の train_battle_seed (全 iter 固定で exploitability を "
                        "iter 間比較可能に)。")
    add_train_side_args(p)
    p.set_defaults(func=psro)

    r = sub.add_parser("rate", parents=[common], help="複数手法の最終レート戦")
    r.add_argument("--funnel-json", type=Path, nargs="+", required=True,
                   help="funnel が出力した *_finalists.json (複数手法 / 多シード可)")
    r.set_defaults(func=rate)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
