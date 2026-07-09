from __future__ import annotations

import argparse
import importlib
import json
import secrets
import time
from pathlib import Path
from typing import Any

from ._native_freshness import check_native_fresh

# _native (Rust) を読む前に .so の鮮度を検査する。古ければここで停止。
check_native_fresh()

from .agent import Agent  # noqa: E402  (鮮度チェック後に import する)
from .encoding import set_mask_opp_obs  # noqa: E402
from .model import default_model_config  # noqa: E402
from .models.modernbert_abs import ModernBertAbsConfig  # noqa: E402
from .supervised import SupervisedConfig, SupervisedStats  # noqa: E402


def build_model_config(
    hidden_size: int | None,
    intermediate_size: int | None,
    num_layers: int | None,
    num_attention_heads: int | None,
) -> ModernBertAbsConfig | None:
    """サイズ系 CLI 引数からモデル設定を作る。いずれも未指定なら None (既定設定)。

    intermediate_size 省略時は hidden_size の 2 倍を既定とする (既定設定 128→256 と同比)。
    max_position_embeddings は default_model_config と同じトークン数に合わせる。
    """
    if all(v is None for v in (hidden_size, intermediate_size, num_layers, num_attention_heads)):
        return None
    base = default_model_config()
    hidden = hidden_size if hidden_size is not None else base.hidden_size
    inter = intermediate_size if intermediate_size is not None else hidden * 2
    layers = num_layers if num_layers is not None else base.num_hidden_layers
    heads = num_attention_heads if num_attention_heads is not None else base.num_attention_heads
    return ModernBertAbsConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        max_position_embeddings=base.max_position_embeddings,
    )


def _format_rate(rate: float) -> str:
    return "n/a" if rate < 0 else f"{rate:.4f}"


def _format_enemy_by(
    enemy_win_by_game: dict[tuple[int, int], tuple[int, int, int]],
    sampler: Any | None,
) -> str:
    """敵 game を敵ラベルで集計し、学習側 P1 の敵別勝率を "exp0=0.53(n=812) ..." 形式で
    返す。キーは (game_id, game_index)。ラベルは sampler の割り当てテーブルから解決し、
    解決後にそのキーを pop してテーブルを進行中ゲームぶんに保つ。"""
    from collections import defaultdict

    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for (gid, gindex), (w, l, d) in enemy_win_by_game.items():
        label = sampler.label_for(gid, gindex) if sampler is not None else None
        if sampler is not None:
            sampler.pop(gid, gindex)
        if label is None:
            label = f"g{gid}"
        a = agg[label]
        a[0] += w
        a[1] += l
        a[2] += d
    parts = []
    for label, (w, l, d) in sorted(agg.items()):
        dec = w + l
        wr = w / dec if dec else 0.0
        parts.append(f"{label}={wr:.3f}(n={dec})")
    return " ".join(parts)


def print_stats(
    stats: SupervisedStats | None,
    enemy_sampler: Any | None = None,
) -> None:
    if stats is None:
        print("update skipped: no examples")
        return
    print(
        "lookahead_update "
        f"examples={stats.examples} "
        f"win_rate={stats.win_rate:.4f} "
        + (f"enemy_win_rate={stats.enemy_win_rate:.4f}(n={stats.enemy_games}) "
           if stats.enemy_games > 0 else "")
        + (f"enemy_by[{_format_enemy_by(stats.enemy_win_by_game, enemy_sampler)}] "
           if stats.enemy_games > 0 and stats.enemy_win_by_game else "")
        + (f"{enemy_sampler.alloc_str()} "
           if enemy_sampler is not None and enemy_sampler.enemies else "")
        +
        f"weakness_move_rate={stats.weakness_move_rate:.4f} "
        f"vs_cloyster_weakness_rate={_format_rate(stats.vs_cloyster_weakness_rate)} "
        f"vs_goodra_weakness_rate={_format_rate(stats.vs_goodra_weakness_rate)} "
        f"switch_prob[model={_format_rate(stats.model_switch_prob)} "
        f"teacher={_format_rate(stats.teacher_switch_prob)}](n={stats.switch_samples}) "
        f"mean_target_value={stats.mean_target_value:.4f} "
        f"policy_loss={stats.policy_loss:.4f} "
        f"value_loss={stats.value_loss:.4f} "
        f"entropy={stats.entropy:.4f} "
        f"raw_logits_std={stats.raw_logits_std:.4f} "
        f"matchup[{_format_matchups(stats.matchup_win_rates)}] "
        f"switch_by_matchup[{_format_matchup_switch(stats.matchup_switch_rates)}]"
    )


def _format_matchups(matchups: dict[str, tuple[float, int]]) -> str:
    # P1視点の各カードを "Cl_v_Go=0.52(n=40)" 形式で並べる。
    return " ".join(
        f"{key}={_format_rate(rate)}(n={n})" for key, (rate, n) in matchups.items()
    )


def _format_matchup_switch(rates: dict[str, tuple[float, float, int]]) -> str:
    # (active技_v_相手種族) ごとの交代確率を "SW_v_Go=m0.75/t0.74(n=80)" 形式で並べる。
    # 攻撃率は 1-交代率。SW_v_Cl=SE / SW_v_Go=半減 / BD_v_Cl=等倍 / BD_v_Go=SE。
    return " ".join(
        f"{key}=m{_format_rate(model_r)}/t{_format_rate(teacher_r)}(n={n})"
        for key, (model_r, teacher_r, n) in rates.items()
    )


def get_rust_async_executor_wrapper() -> Any:
    # 実行時ビルドはしない。.so の鮮度は import 前に check_native_fresh() が検査済みで、
    # ビルドは Makefile の `maturin develop --release` が保証する。ここで `maturin develop`
    # (--release なし=debug) を走らせると、ロード済みの release .so をディスク上で debug に
    # 上書きしてしまい、次プロセス以降が debug .so を掴んで激遅になる事故が起きる。
    module = importlib.import_module("poke_ai3")
    return module.get_rust_async_executor_wrapper


class TrainSession:
    """学習の executor + Agent を 1 回だけ構築し、`run_to(target_epoch)` で前進させ続ける
    常駐セッション。ckpt_tournament の funnel がブロックごとにサブプロセスを起動していた
    固定費 (uv 再ビルド判定 + torch import + Agent ロード + CUDA Graph 構築) を排すため、
    executor / agent をインスタンスに保持してブロックをまたいで使い回す。

    executor は 1 回だけ battle_seed を決めて構築し連続生成する (旧サブプロセス方式は
    ブロックごとに新 seed だったが、常駐化により run 全体で連続した RNG ストリームになる)。"""

    def __init__(
        self,
        num_games: int,
        max_batch_size: int | None,
        trajectories_threshold: int | None,
        sleep_seconds: float,
        device: str | None,
        checkpoint_path: Path | None,
        backend: str,
        randomize: bool,
        crit_enabled: bool,
        stage: str,
        sims: int,
        sim_concurrency: int,
        search_turn_min: int,
        search_turn_max: int,
        depth_skew: float,
        battle_seed: int | None,
        mask_opp_obs: bool = False,
        infer_graph: bool = True,
        nash_weak: bool = True,
        nash_learning_rate: float = 1.5,
        value_target_expected: bool = False,
        learning_rate: float | None = None,
        model_config: ModernBertAbsConfig | None = None,
        minibatch_size: int | None = None,
        supervised_epochs: int | None = None,
    ) -> None:
        if mask_opp_obs:
            set_mask_opp_obs(True)
            print("mask_opp_obs: enabled (baseline; opponent extended observation zeroed)")
        if battle_seed is None:
            battle_seed = secrets.randbits(64)
        print(f"battle_seed: {battle_seed}")
        # 推論バッチ上限の既定は「同時 in-flight 推論の理論上限 (num_games*sim_concurrency*2)
        # の 3/7」。experiments/poke-ai3 20260703 で 12/36 epoch のバッチ掃引を行い、2/5〜3/5 は
        # ノイズ帯で横並び・target-cpu=native も無効と確認した上で 3/7 を代表値に採用。
        if max_batch_size is None:
            max_batch_size = round(num_games * sim_concurrency * 2 * 3 / 7)
            print(f"max_batch_size: {max_batch_size} (既定 num_games*sim_concurrency*2*3/7)")
        self.sleep_seconds = sleep_seconds
        self.checkpoint_path = checkpoint_path
        self.num_games = num_games
        # 敵混合学習用の推論フック。None なら単一モデル (agent.infer_step)。funnel が
        # プール推論 (学習者 + 凍結敵を game_id ルーティング) を注入するとそれを使う。
        self.infer_fn: Any | None = None
        # per-game 敵割り当て器 (EnemySampler)。configure_enemies が設定する。None なら
        # 単一モデル自己対戦で、enemy_by / alloc ログは出さない。
        self.enemy_sampler: Any | None = None
        self.executor = get_rust_async_executor_wrapper()(
            num_games,
            max_batch_size,
            trajectories_threshold,
            backend,
            randomize,
            crit_enabled,
            stage,
            sims,
            sim_concurrency,
            search_turn_min,
            search_turn_max,
            depth_skew=depth_skew,
            battle_seed=battle_seed,
            nash_learning_rate=nash_learning_rate,
            nash_weak=nash_weak,
            value_target_expected=value_target_expected,
        )
        if value_target_expected:
            print("value_target: expected (均衡混合 training_pi による期待勝率)")
        if nash_weak:
            print(f"nash_weak: enabled (穏当化版, nash_learning_rate={nash_learning_rate})")
        # 既存 checkpoint から再開する場合は保存済みの設定を優先する (サイズ不一致での
        # load 失敗を防ぐ)。新規学習のときだけ CLI 由来の model_config を使う。
        if model_config is not None and checkpoint_path is not None and checkpoint_path.exists():
            print(
                "warning: 既存 checkpoint から再開するため --hidden-size 等のサイズ指定は無視し、"
                "checkpoint に保存された設定を使います。",
                flush=True,
            )
            model_config = None
        if model_config is not None:
            print(
                f"model_config: hidden={model_config.hidden_size} "
                f"intermediate={model_config.intermediate_size} "
                f"layers={model_config.num_hidden_layers} "
                f"heads={model_config.num_attention_heads}"
            )
        supervised_config = None
        if minibatch_size is not None or supervised_epochs is not None:
            defaults = SupervisedConfig()
            supervised_config = SupervisedConfig(
                epochs=supervised_epochs if supervised_epochs is not None else defaults.epochs,
                minibatch_size=(
                    minibatch_size if minibatch_size is not None else defaults.minibatch_size
                ),
            )
            print(
                f"supervised_config: epochs={supervised_config.epochs} "
                f"minibatch_size={supervised_config.minibatch_size}"
            )
        self.agent = Agent(
            device=device,
            checkpoint_path=checkpoint_path,
            infer_graph=infer_graph,
            learning_rate=learning_rate,
            model_config=model_config,
            supervised_config=supervised_config,
            infer_max_batch_size=max_batch_size,
        )
        # checkpoint から再開した場合は training_step からエポック番号を継続する
        # (target_epochs は通算エポック数。例: ep30 から再開し target=40 で 31..40)。
        self.epochs = self.agent.training_step

    def run_to(
        self,
        target_epochs: int | None,
        snapshot_every: int | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        """self.epochs が target_epochs に達するまで学習を進める。
        到達済みなら即 return (no-op)。checkpoint_path 省略時は構築時の値を使う。"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        executor = self.executor
        agent = self.agent
        while target_epochs is None or self.epochs < target_epochs:
            if executor.trajectories_ready():
                py_obj_with_trajectories = json.loads(executor.recv_trajectories())
                stats = agent.learn(py_obj_with_trajectories)
                print_stats(stats, self.enemy_sampler)
                self.epochs += 1
                # 10ep ごと等のスナップショット保存 (勝率曲線を後で eval するため)。
                # checkpoint_path=foo.pt なら foo_ep10.pt のように epoch 番号を挟む。
                if (
                    snapshot_every
                    and checkpoint_path is not None
                    and stats is not None
                    and self.epochs % snapshot_every == 0
                ):
                    snap = checkpoint_path.with_name(
                        f"{checkpoint_path.stem}_ep{self.epochs}{checkpoint_path.suffix}"
                    )
                    agent.save_checkpoint(stats, path=snap)
                    print(f"snapshot saved: {snap}", flush=True)
            elif executor.is_ready():
                if self.infer_fn is not None:
                    self.infer_fn(executor)
                else:
                    agent.infer_step(executor)
            else:
                time.sleep(self.sleep_seconds)


def run_train_loop(
    num_games: int,
    max_batch_size: int | None,
    trajectories_threshold: int | None,
    max_epochs: int | None,
    sleep_seconds: float,
    device: str | None,
    checkpoint_path: Path | None,
    backend: str,
    randomize: bool,
    crit_enabled: bool,
    stage: str,
    sims: int,
    sim_concurrency: int,
    search_turn_min: int,
    search_turn_max: int,
    depth_skew: float,
    battle_seed: int | None,
    mask_opp_obs: bool = False,
    infer_graph: bool = True,
    snapshot_every: int | None = None,
    nash_weak: bool = True,
    nash_learning_rate: float = 1.5,
    value_target_expected: bool = False,
    learning_rate: float | None = None,
    model_config: ModernBertAbsConfig | None = None,
    minibatch_size: int | None = None,
    supervised_epochs: int | None = None,
) -> None:
    session = TrainSession(
        num_games=num_games,
        max_batch_size=max_batch_size,
        trajectories_threshold=trajectories_threshold,
        sleep_seconds=sleep_seconds,
        device=device,
        checkpoint_path=checkpoint_path,
        backend=backend,
        randomize=randomize,
        crit_enabled=crit_enabled,
        stage=stage,
        sims=sims,
        sim_concurrency=sim_concurrency,
        search_turn_min=search_turn_min,
        search_turn_max=search_turn_max,
        depth_skew=depth_skew,
        battle_seed=battle_seed,
        mask_opp_obs=mask_opp_obs,
        infer_graph=infer_graph,
        nash_weak=nash_weak,
        nash_learning_rate=nash_learning_rate,
        value_target_expected=value_target_expected,
        learning_rate=learning_rate,
        model_config=model_config,
        minibatch_size=minibatch_size,
        supervised_epochs=supervised_epochs,
    )
    session.run_to(max_epochs, snapshot_every=snapshot_every, checkpoint_path=checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=64)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help=(
            "推論バッチの上限。同時 in-flight 推論数の理論上限は "
            "num-games * sim-concurrency * 2 なので、これを超える値を指定すると"
            "バッチが永遠に埋まらず生成が停滞する。推奨・既定はこの理論上限の 3/7 "
            "(未指定なら num-games*sim-concurrency*2*3/7 を自動算出。既定 num-games 64 * "
            "sim-concurrency 16 なら 878。experiments/poke-ai3 20260703 のバッチ掃引で確定)。"
        ),
    )
    # 既定 128 は num-games 64 系の learn/save 割り込み頻度を抑えるスループット最適値
    # (experiments/poke-ai3 20260625 step1-2)。num-games を大きく下げる場合は要調整。
    parser.add_argument("--trajectories-threshold", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=None,
        help="N エポックごとに checkpoint_path に epoch 番号を挟んだスナップショットを"
        "保存する (例: ckpt.pt → ckpt_ep10.pt)。勝率曲線の eval 用。--checkpoint-path 必須。",
    )
    # 50ms だと「バッチ到着の谷で sleep に入り全ゲームが推論待ちで止まる」準安定
    # スローダウンを起こす (experiments/poke-ai3 20260612 参照)。0 (GIL を放して
    # 即戻るビジーポーリング) が最速で、1ms 比で約 5% 上。
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Load from this checkpoint if present, and save updates back to it.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "showdown"],
        default="local",
        help="Battle backend: 'local' (poke-sho-rust in-process) or 'showdown' (Pokemon Showdown subprocess).",
    )
    parser.add_argument(
        "--random",
        dest="randomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the 16-step damage roll (local backend). "
            "決定論モードは --no-random、ロール有効化は --random。"
        ),
    )
    parser.add_argument(
        "--crit",
        dest="crit_enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable critical hits (local backend). 決定論評価では --no-crit。",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["3a", "3b", "3c"],
        default="3b",
        help="シナリオの難易度ステージ: "
        "3a (タイプ相性導入・4技1v1, Cloyster vs Goodra-Hisui) / "
        "3b (交代学習・非対称2チーム・各個体1技) / "
        "3c (対称対面の交代学習・FightSpe60/FairyPhy60・各個体1技).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=64,
        help="lookahead の 1 局面あたり rollout 回数。実行速度に応じて調整する。",
    )
    parser.add_argument(
        "--sim-concurrency",
        type=int,
        default=16,
        help=(
            "1 局面の lookahead 内で同時に走らせる rollout 本数 (スライディングウィンドウ幅)。"
            "1 で従来の逐次挙動。--sims 以下である必要がある。1 試合でも GPU を飽和させたい"
            "ときに増やす。適切な値は実験で決める。"
        ),
    )
    parser.add_argument(
        "--search-turn-min",
        type=int,
        default=4,
        help="rollout の最小深さ (ply)。終局はこれより手前で来ることが多い。",
    )
    parser.add_argument(
        "--search-turn-max",
        type=int,
        default=8,
        help="rollout の最大深さ (ply)。終局しない場合はここで value net 評価に切り替える。",
    )
    parser.add_argument(
        "--depth-skew",
        type=float,
        default=1.0,
        help=(
            "depth_cap を [min..=max] から選ぶときの深側への偏り。深さ k 番目の重みを "
            "depth_skew^k とする。1.0 で一様 (従来挙動)、2.0 で1手深いごとに確率2倍。"
            "行動分布は変えず value 推定の打ち切り地平線の配分のみ変える (Nash-safe)。"
        ),
    )
    parser.add_argument(
        "--mask-opp-obs",
        dest="mask_opp_obs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "相手側拡張観測 (相手 active 技・相手控え) をゼロ化する。"
            "観測拡張 A/B のベースライン (旧観測相当・アーキテクチャ同一) 用。"
        ),
    )
    parser.add_argument(
        "--infer-graph",
        dest="infer_graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "推論の forward+softmax を CUDA Graphs (バッチをバケットにパディングして "
            "1 replay) にまとめる。--no-infer-graph で従来の逐次発行 (autocast) に戻す。"
        ),
    )
    parser.add_argument(
        "--battle-seed",
        type=int,
        default=None,
        help=(
            "対戦生成 (シミュレータ乱数・対戦シード列) のシード。省略時はランダム生成し"
            "起動時にログ出力する。同一シードでも GPU 推論と並列実行のタイミングにより"
            "学習全体のビット単位の再現は保証されない。"
        ),
    )
    parser.add_argument(
        "--nash-weak",
        dest="nash_weak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "nash_accumulation の穏当化版 (崖なし) を使う。平均勝率の半分以下の手を "
            "training 0 に落とさず、係数を 1.0 中心の乗数 [1/lr, lr] に圧縮する。"
            "2 seed の A/B で strict 版と同等以上だったためデフォルト有効。"
            "--no-nash-weak で strict 版 (崖あり) に戻せる (様子見で残置、採用機会が無ければ削除予定)。"
        ),
    )
    parser.add_argument(
        "--nash-learning-rate",
        type=float,
        default=1.5,
        help="nash accumulation の learning rate (係数の鋭さ)。穏当化版では分布の鋭さを直接制御する。",
    )
    parser.add_argument(
        "--value-target",
        dest="value_target",
        choices=["max", "expected"],
        default="max",
        help=(
            "value 教師の式。max (既定) は手ごと最大勝率 max_i win_rates[i]。"
            "expected は均衡混合 Σ_i training_pi[i]*win_rates[i]。max はゼロサム同時手番で "
            "均衡値以上へ出る (勝率過大評価) ため expected で較正する A/B 用。"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "オプティマイザ (AdamW) の learning rate。省略時は checkpoint 値を尊重 "
            "(新規学習は 1e-4)。明示時は checkpoint 由来の lr を上書きする (lr スイープ用)。"
        ),
    )
    # モデルサイズ系 (新規学習時のみ有効。既存 checkpoint 再開時は無視され保存設定を使う)。
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="モデルの hidden_size (既定 128)。num-attention-heads で割り切れる必要がある。",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=None,
        help="MLP の intermediate_size。省略時は hidden_size の 2 倍。",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="エンコーダ層数 num_hidden_layers (既定 2)。",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=None,
        help="注意ヘッド数 (既定 4)。hidden_size を割り切る必要がある。",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=256,
        help="教師あり学習のミニバッチサイズ。大きくすると learn ステップの GPU 稼働率が "
        "上がり、エポック境界の GPU 使用率低下が浅くなる。既定 256 は examples/s 1.69x を "
        "強さ非劣化で確定した値 (experiments/poke-ai3 20260625 step3)。",
    )
    parser.add_argument(
        "--supervised-epochs",
        type=int,
        default=None,
        help="1 epoch (1 バッチの生成データ) を教師あり学習で何パスなめるか。"
        "毎パスでシャッフルし直す。未指定なら既定 (SupervisedConfig.epochs=4)。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_train_loop(
        num_games=args.num_games,
        max_batch_size=args.max_batch_size,
        trajectories_threshold=args.trajectories_threshold,
        max_epochs=args.max_epochs,
        sleep_seconds=args.sleep_seconds,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        backend=args.backend,
        randomize=args.randomize,
        crit_enabled=args.crit_enabled,
        stage=args.stage,
        sims=args.sims,
        sim_concurrency=args.sim_concurrency,
        search_turn_min=args.search_turn_min,
        search_turn_max=args.search_turn_max,
        depth_skew=args.depth_skew,
        battle_seed=args.battle_seed,
        mask_opp_obs=args.mask_opp_obs,
        infer_graph=args.infer_graph,
        snapshot_every=args.snapshot_every,
        nash_weak=args.nash_weak,
        nash_learning_rate=args.nash_learning_rate,
        value_target_expected=(args.value_target == "expected"),
        learning_rate=args.learning_rate,
        model_config=build_model_config(
            args.hidden_size,
            args.intermediate_size,
            args.num_layers,
            args.num_attention_heads,
        ),
        minibatch_size=args.minibatch_size,
        supervised_epochs=args.supervised_epochs,
    )


if __name__ == "__main__":
    main()
