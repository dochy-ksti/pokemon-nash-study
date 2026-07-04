"""HP 局面別 (5×5 unfold) の交代戦略を観察する専用スクリプト (stage3c の主役指標)。

学習ループ内の毎エポック診断は軽量な対面別指標に留め、本スクリプトは収束した
checkpoint を固定して大量に self-play を回し、交代確率を
「自分 active HP バケット × 相手 active HP バケット」の 5×5 で層別して表にする。

ねらい: 3c で「タイプ非対称を除いても 3HKO 由来の HP 状況依存の混合戦略が残るか」を
定量確認する。鏡像セルは畳まず (unfold)、3b と 3c の両 checkpoint に同じものを当てて
並べることで「3b=非対称 / 3c=対称」の構造差を読む。サンプルの枯れたセルは最小 n ガードで
「観察不能 (--)」と正直に表示し、確率を主張しない。

起動例:
  uv run eval-hp-strategy --checkpoint data/poke-ai3/stage3c_weak_s1.pt --stage 3c \\
      --num-games 32 --num-eval-games 4000 --sims 64 --search-turn-min 6 --search-turn-max 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .agent import Agent
from .diagnostics import (
    HP_BUCKET_LABELS,
    SWITCH_MATCHUP_KEYS,
    hp_stratified_switch_diagnostics,
)
from .encoding import encode_observations
from .train_loop import get_rust_async_executor_wrapper


def collect_examples(
    executor: Any,
    agent: Agent,
    num_games_target: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    """self-play を回し、num_games_target 試合ぶんの trajectory item を集めて返す。"""
    examples: list[dict[str, Any]] = []
    games = 0
    while games < num_games_target:
        if executor.trajectories_ready():
            payload = json.loads(executor.recv_trajectories())
            for trajectory in payload.get("vec", []):
                examples.extend(trajectory.get("items", []))
                games += 1
            print(f"  games so far: {games}/{num_games_target}  (items={len(examples)})")
        elif executor.is_ready():
            agent.infer_step(executor)
        else:
            time.sleep(sleep_seconds)
    return examples


def _print_grid(key: str, cells: dict[tuple[str, int, int], tuple[float, float, int]], min_n: int) -> None:
    """1 対面キーの 5×5 グリッドを表示する。行=自分 active HP, 列=相手 active HP。
    各セルは model の交代確率 (n<min_n は '--')。該当キーのデータが皆無なら見出しのみ。"""
    present = {(my_b, opp_b): v for (k, my_b, opp_b), v in cells.items() if k == key}
    total_n = sum(v[2] for v in present.values())
    print(f"\n=== {key}  (n={total_n}) ===")
    if total_n == 0:
        print("  (データなし)")
        return
    header = "  自\\相 | " + " ".join(f"{lab:>6}" for lab in HP_BUCKET_LABELS)
    print(header)
    for my_b in range(5):
        row_cells = []
        for opp_b in range(5):
            v = present.get((my_b, opp_b))
            if v is None or v[2] < min_n:
                cell = "--" if v is None else f"--({v[2]})"
                row_cells.append(f"{cell:>6}")
            else:
                row_cells.append(f"{v[0]:>6.2f}")
        print(f"  {HP_BUCKET_LABELS[my_b]:>5} | " + " ".join(row_cells))
    # 参考: teacher (lookahead) 側の交代確率も総計で出す。
    msum = sum(v[0] * v[2] for v in present.values())
    tsum = sum(v[1] * v[2] for v in present.values())
    print(f"  集計交代確率: model={msum / total_n:.3f}  teacher={tsum / total_n:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", type=str, choices=["3a", "3b", "3c"], default="3c")
    parser.add_argument("--num-games", type=int, default=32, help="並列ゲーム数 (Rust executor)。")
    parser.add_argument("--num-eval-games", type=int, default=4000,
                        help="集計する総試合数。5×5 セルを埋めるため多めに。")
    parser.add_argument("--min-n", type=int, default=30,
                        help="セルを表示する最小サンプル数。下回るセルは観察不能 (--) 扱い。")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--trajectories-threshold", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=["local", "showdown"], default="local")
    parser.add_argument("--random", dest="randomize",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crit", dest="crit_enabled",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--sim-concurrency", type=int, default=1)
    parser.add_argument("--search-turn-min", type=int, default=6)
    parser.add_argument("--search-turn-max", type=int, default=12)
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
        False,  # eval_rule_opponent: self-play (両側 NN)
        False,  # eval_rule_p1
    )
    agent = Agent(device=args.device, checkpoint_path=args.checkpoint)
    print(f"checkpoint={args.checkpoint}  stage={args.stage}  "
          f"num_eval_games={args.num_eval_games}  min_n={args.min_n}")

    examples = collect_examples(executor, agent, args.num_eval_games, args.sleep_seconds)
    if not examples:
        print("交代局面サンプルが集まりませんでした。")
        return

    encoded = encode_observations(examples, agent.device)
    cells = hp_stratified_switch_diagnostics(
        agent.model, encoded, examples, agent.device, agent.agent_config.amp_dtype
    )

    print(f"\n総 item 数={len(examples)}  交代合法かつ対面分類できたセル数={len(cells)}")
    print("行=自分 active HP バケット / 列=相手 active HP バケット / 値=model の交代確率")
    print("(HP バケット: 100=満タン / >=75 / >=50 / >=25 / <25。unfold=鏡像セルは別集計)")
    for key in SWITCH_MATCHUP_KEYS:
        _print_grid(key, cells, args.min_n)


if __name__ == "__main__":
    main()
