"""2 つの学習済み checkpoint を直接対戦させ、head-to-head 勝率を測る評価ツール。

固定ルール eval (eval_vs_rule) は「weakness_move_rate=100% の決定論ルール AI」が相手で、
それを最大限スポイルできる方策と、混合戦略ナッシュ均衡に近い「真に強い」方策は必ずしも
一致しない。本ツールは rule AI を介さず、checkpoint A (P1) と checkpoint B (P2) を
直接戦わせることで「eval 勝率が下がったが学習が進んだモデル」と「学習は浅いが eval 勝率が
高いモデル」のどちらが実際に強いかを切り分ける。

実装: self-play executor (両側 NN 推論) を使い、infer_step で観測バッチを player で分割し、
P1 (player==0) は agent_a、P2 (player==1) は agent_b にルーティングする。学習はしない。
先発の side 有利を相殺するため、--swap を付けて A/B の P1/P2 を入れ替えた 2 回を回すこと。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .agent import Agent
from .train_loop import get_rust_async_executor_wrapper
from poke_ai3 import ACTION_DIM

# infer_encoded が参照する観測テンソルのキー (packed ホットパス / フィールド別 dict)。
# player 分割時はこれらだけを行スライスすれば十分 (empty_* echo 列は対象外)。
_PACKED_KEYS = ("packed_i64", "packed_f32", "legal_action_mask")


def _slice_obs(obs: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    """packed 観測を行 index で切り出した sub-obs を作る。

    fancy indexing は contiguous な copy を返すため torch.from_numpy にそのまま渡せる。"""
    keys = _PACKED_KEYS if "packed_i64" in obs else None
    if keys is None:
        # フィールド別 dict (診断経路)。先頭次元がバッチのものだけ切り出す。
        return {
            k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == obs["player"].shape
                else v)
            for k, v in obs.items()
        }
    return {k: obs[k][idx] for k in keys}


def infer_step_split(executor: Any, agent_a: Agent, agent_b: Agent) -> None:
    """観測バッチを player で分割し、P1=agent_a / P2=agent_b で推論して返送する。

    P1 行は agent_a、P2 行は agent_b だけに forward する (旧実装の「両 agent にフル
    バッチを通して mask で採用」する 2x 演算を廃し、有効行数を 2B→B に削減)。
    結果は元の行順へ scatter して返す。"""
    obs = executor.recv_observations()
    player = obs["player"]
    n = int(player.shape[0])
    if n == 0:
        # 空バッチ。echo だけ返す。
        executor.send_inference(
            obs["game_id"], obs["player"], obs["request_id"],
            np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32),
            obs["empty_game_id"], obs["empty_player"], obs["empty_request_id"],
        )
        return
    policy = np.empty((n, ACTION_DIM), dtype=np.float32)
    value = np.empty((n,), dtype=np.float32)
    for agent, pid in ((agent_a, 0), (agent_b, 1)):
        idx = np.nonzero(player == pid)[0]
        if idx.size == 0:
            continue
        pol, val = agent.infer_encoded(_slice_obs(obs, idx))
        policy[idx] = pol
        value[idx] = val
    executor.send_inference(
        obs["game_id"],
        obs["player"],
        obs["request_id"],
        np.ascontiguousarray(policy, dtype=np.float32).ravel(),
        np.ascontiguousarray(value, dtype=np.float32),
        obs["empty_game_id"],
        obs["empty_player"],
        obs["empty_request_id"],
    )


class EnemySampler:
    """敵混合学習の per-game 敵割り当て。

    敵は「(game_id, game_index) 単位で σ 比に配分」する。旧実装は game_id スロットへ敵を
    固定していたため、実効ゲーム数シェアが「σ ÷ 平均ゲーム長」に歪む (短い試合の敵ほど
    速く回って過剰露出) バグがあった。ここでは各ゲーム開始 (新しい game_index) を鍵に、
    不足カウンタ (Hamilton ストリーミング: w_i*(total+1) - assigned_i が最大の敵を選ぶ)
    で 1 体を割り当てる。累積シェアは常に σ から ±1 ゲーム以内に張り付き、ゲーム長に依らない。

    同一ゲーム中 (同じ game_index の連続手番) は割り当てを固定。trajectory 消費時に pop して
    テーブルを進行中ゲームぶん (≦ num_games) に保つ。deficit カウンタ (assigned/total) は
    iter 全体で累積し pop では減らさない。"""

    def __init__(
        self, enemies: list[Agent], labels: list[str],
        weights: list[float], enemy_gids: set[int],
    ) -> None:
        self.enemies = enemies
        self.labels = labels
        w = np.asarray(weights, dtype=float)
        self.w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / max(len(w), 1)
        self.enemy_gids = {int(g) for g in enemy_gids}
        self.assigned = np.zeros(len(enemies), dtype=float)
        self.total = 0
        self.table: dict[tuple[int, int], int] = {}

    def idx_for(self, gid: int, gindex: int) -> int:
        key = (gid, gindex)
        idx = self.table.get(key)
        if idx is None:
            idx = int(np.argmax(self.w * (self.total + 1) - self.assigned))
            self.table[key] = idx
            self.assigned[idx] += 1.0
            self.total += 1
        return idx

    def label_for(self, gid: int, gindex: int) -> str | None:
        idx = self.table.get((gid, gindex))
        return self.labels[idx] if idx is not None else None

    def pop(self, gid: int, gindex: int) -> None:
        self.table.pop((gid, gindex), None)

    def alloc_str(self) -> str:
        """実効割り当てシェア vs σ を "exp0=.41/σ.41 ..." で返す (露出が σ 通りか検証用)。"""
        if self.total == 0:
            return "alloc[none]"
        share = self.assigned / self.total
        parts = [f"{self.labels[i]}={share[i]:.2f}/σ{self.w[i]:.2f}"
                 for i in range(len(self.enemies))]
        return f"alloc[{' '.join(parts)}](n={self.total})"


def infer_step_pool(
    executor: Any, learner: Agent, router: EnemySampler | None,
) -> None:
    """敵混合学習用の推論ステップ。(game_id, game_index, player) で担当エージェントへ
    ルーティングする。

    - P1 行 (player==0) は常に学習者 learner。
    - P2 行 (player==1) は、その game_id が敵スロットなら router が (game_id, game_index) 単位で
      σ 配分して選んだ凍結敵、そうでなければ (自己対戦) learner。
    router が None のときは全行 learner (agent.infer_step と等価)。
    同一敵に割り当たった行はまとめて 1 回 forward する。"""
    obs = executor.recv_observations()
    player = obs["player"]
    n = int(player.shape[0])
    if n == 0:
        executor.send_inference(
            obs["game_id"], obs["player"], obs["request_id"],
            np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32),
            obs["empty_game_id"], obs["empty_player"], obs["empty_request_id"],
        )
        return
    policy = np.empty((n, ACTION_DIM), dtype=np.float32)
    value = np.empty((n,), dtype=np.float32)
    learner_mask = np.ones(n, dtype=bool)
    if router is not None and router.enemies:
        game_id = obs["game_id"]
        game_index = obs["game_index"]
        # P2 かつ敵スロットの行に、(game_id, game_index) 単位で敵 idx を割り当てる。
        enemy_of = np.full(n, -1, dtype=np.int64)
        for r in np.nonzero(player == 1)[0]:
            gid = int(game_id[r])
            if gid in router.enemy_gids:
                enemy_of[r] = router.idx_for(gid, int(game_index[r]))
        for e in range(len(router.enemies)):
            idx = np.nonzero(enemy_of == e)[0]
            if idx.size == 0:
                continue
            pol, val = router.enemies[e].infer_encoded(_slice_obs(obs, idx))
            policy[idx] = pol
            value[idx] = val
            learner_mask[idx] = False
    lidx = np.nonzero(learner_mask)[0]
    if lidx.size:
        pol, val = learner.infer_encoded(_slice_obs(obs, lidx))
        policy[lidx] = pol
        value[lidx] = val
    executor.send_inference(
        obs["game_id"],
        obs["player"],
        obs["request_id"],
        np.ascontiguousarray(policy, dtype=np.float32).ravel(),
        np.ascontiguousarray(value, dtype=np.float32),
        obs["empty_game_id"],
        obs["empty_player"],
        obs["empty_request_id"],
    )


def collect_results(
    executor: Any,
    agent_a: Agent,
    agent_b: Agent,
    num_games_target: int,
    sleep_seconds: float,
    print_every: int = 0,
    quiet_progress: bool = False,
) -> list[int]:
    """num_games_target 実バトルぶんの結果を集め、[A勝(P1), B勝(P2), 引分] を計上する。

    1 実バトルは P1/P2 の 2 trajectory を生み、両者は同じ winner を持つ。実試合数を
    正しく数えるため P1 側 trajectory だけを計上する (player フィルタは items 空でも壊れず、
    ルール対戦で片側のみ送られる経路でも正しい)。
    """
    tally = [0, 0, 0]
    games = 0
    last_print = 0
    while games < num_games_target:
        if executor.trajectories_ready():
            payload = json.loads(executor.recv_trajectories())
            for trajectory in payload.get("vec", []):
                if str(trajectory.get("player")) != "P1":
                    continue
                winner = str(trajectory.get("winner"))
                idx = 0 if winner == "P1" else (1 if winner == "P2" else 2)
                tally[idx] += 1
                games += 1
            if not quiet_progress and print_every > 0 and games - last_print >= print_every:
                last_print = games
                print(f"  games so far: {games}/{num_games_target}")
        elif executor.is_ready():
            infer_step_split(executor, agent_a, agent_b)
        else:
            time.sleep(sleep_seconds)
    return tally


def _fmt(counts: list[int], a_label: str, b_label: str) -> str:
    a_win, b_win, draw = counts
    n = a_win + b_win + draw
    if n == 0:
        return "  n/a (n=0)"
    return (f"{a_label}(P1) 勝率={a_win / n:.3f}  "
            f"(A={a_win} B={b_win} D={draw}, n={n})  {b_label}=P2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True, help="P1 側 checkpoint")
    parser.add_argument("--checkpoint-b", type=Path, required=True, help="P2 側 checkpoint")
    parser.add_argument("--num-games", type=int, default=256,
                        help="並列ゲーム数 (Rust executor)。head-to-head は学習が無いため "
                             "並列度を上げて batch を太らせるのが有利 (Phase3 既定 256)。"
                             "GPU充填には --max-batch-size を num-games に揃えるとよい。")
    parser.add_argument("--num-eval-games", type=int, default=512, help="集計する総試合数。")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--trajectories-threshold", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0,
                        help="idle 時 sleep 秒。学習しない head-to-head は既定 0 (完全スピン) "
                             "で 50ms 谷を除去。1 台で複数評価を並列する場合のみ正値に。")
    parser.add_argument("--quiet-progress", dest="quiet_progress",
                        action="store_true", default=False,
                        help="途中の games so far 表示を抑止 (rate のログ整理用)。")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=["local", "showdown"], default="local")
    parser.add_argument("--random", dest="randomize",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crit", dest="crit_enabled",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stage", type=str, choices=["3a", "3b", "3c"], default="3b")
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--sim-concurrency", type=int, default=1)
    parser.add_argument("--search-turn-min", type=int, default=4)
    parser.add_argument("--search-turn-max", type=int, default=8)
    parser.add_argument(
        "--policy-only", dest="policy_only",
        action=argparse.BooleanOptionalAction, default=False,
        help="lookahead を廃し policy net 単発推論で確率着手する高速対戦モード。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executor = get_rust_async_executor_wrapper()(
        args.num_games,
        args.max_batch_size,
        args.trajectories_threshold,
        args.backend,
        args.randomize,
        args.crit_enabled,
        args.stage,
        args.sims,
        args.sim_concurrency,
        args.search_turn_min,
        args.search_turn_max,
        False,  # eval_rule_opponent: P2 も NN にする (self-play 経路)
        False,  # eval_rule_p1
        policy_only=args.policy_only,
    )
    agent_a = Agent(device=args.device, checkpoint_path=args.checkpoint_a)
    agent_b = Agent(device=args.device, checkpoint_path=args.checkpoint_b)
    a_label = args.checkpoint_a.stem
    b_label = args.checkpoint_b.stem
    print(f"A(P1)={args.checkpoint_a} step={agent_a.training_step}")
    print(f"B(P2)={args.checkpoint_b} step={agent_b.training_step}")
    print(f"stage={args.stage} num_games={args.num_games} num_eval_games={args.num_eval_games}")
    tally = collect_results(
        executor, agent_a, agent_b, args.num_eval_games, args.sleep_seconds,
        print_every=args.num_games, quiet_progress=args.quiet_progress,
    )
    print(f"\n=== head-to-head: {a_label} vs {b_label} ===")
    print(f"  {_fmt(tally, a_label, b_label)}")
    # 機械可読 (orchestrator 用): a_win b_win draw
    print(f"RESULT a_win={tally[0]} b_win={tally[1]} draw={tally[2]}", flush=True)


if __name__ == "__main__":
    main()
