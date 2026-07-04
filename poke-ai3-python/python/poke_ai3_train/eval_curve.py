"""複数チェックポイント (例: 10ep ごとのスナップショット) を順に固定ルール AI と対戦させ、
勝率の推移 (学習曲線) を出力する評価ツール。

各チェックポイントは fresh な executor + Agent で評価する。executor を使い回すと
重み切替時にゲームが旧重みで途中まで進み計測が汚染されるため、スナップショットごとに
executor を作り直す。Agent は checkpoint の model_config から自動でモデルを再構築する
(hidden_size 違いの 128/256 を同じスクリプトで扱える)。

使い方:
  uv run eval-curve --checkpoint-glob 'data/poke-ai3/ckpt_h128_ep*.pt' \
    --num-games 512 --sim-concurrency 16 --num-eval-games 512 --sims 64 \
    --search-turn-min 6 --search-turn-max 12 --no-random --no-crit --stage 3b
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from .agent import Agent
from .encoding import set_mask_opp_obs
from .eval_vs_rule import collect_results
from .diagnostics import SWITCH_MATCHUP_KEYS
from .train_loop import get_rust_async_executor_wrapper

_SE_MATCHUPS = {"SW_v_Cl", "BD_v_Go", "FS_v_Cl", "FP_v_Go"}


def _epoch_of(path: Path) -> int:
    """ファイル名末尾の _ep<N> から epoch を取り出す。無ければ大きな値 (末尾) 扱い。"""
    m = re.search(r"_ep(\d+)", path.stem)
    return int(m.group(1)) if m else 1 << 30


def _win_rate(counts: list[int]) -> tuple[float, int, int, int]:
    win, loss, draw = counts
    n = win + loss + draw
    return (win / n if n else 0.0, win, loss, draw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-glob", type=str, required=True,
                        help="評価するチェックポイント群の glob (例 'data/poke-ai3/ckpt_h128_ep*.pt')。")
    parser.add_argument("--num-games", type=int, default=512)
    parser.add_argument("--num-eval-games", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--trajectories-threshold", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=["local", "showdown"], default="local")
    parser.add_argument("--random", dest="randomize",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crit", dest="crit_enabled",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stage", type=str, choices=["3a", "3b", "3c"], default="3b")
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--sim-concurrency", type=int, default=16)
    parser.add_argument("--search-turn-min", type=int, default=4)
    parser.add_argument("--search-turn-max", type=int, default=8)
    parser.add_argument("--mask-opp-obs", dest="mask_opp_obs",
                        action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mask_opp_obs:
        set_mask_opp_obs(True)
    paths = sorted((Path(p) for p in glob.glob(args.checkpoint_glob)), key=_epoch_of)
    if not paths:
        print(f"no checkpoints matched: {args.checkpoint_glob}")
        return
    make_executor = get_rust_async_executor_wrapper()
    print(f"eval-curve: {len(paths)} checkpoints, num_eval_games={args.num_eval_games}")
    rows: list[tuple[int, float, float, float, int]] = []
    for path in paths:
        ep = _epoch_of(path)
        # スナップショットごとに executor を作り直す (旧重み汚染を避ける)。
        executor = make_executor(
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
            True,   # eval_rule_opponent: P2 を固定ルールにする
            False,  # eval_rule_p1
        )
        agent = Agent(device=args.device, checkpoint_path=path)
        tally = collect_results(executor, agent, args.num_eval_games, args.sleep_seconds)
        all_wr, w, l, d = _win_rate(tally["ALL"])
        fav = [sum(x) for x in zip(*(tally[k] for k in SWITCH_MATCHUP_KEYS if k in _SE_MATCHUPS))]
        unf = [sum(x) for x in zip(*(tally[k] for k in SWITCH_MATCHUP_KEYS if k not in _SE_MATCHUPS))]
        fav_wr = _win_rate(fav)[0]
        unf_wr = _win_rate(unf)[0]
        rows.append((ep, all_wr, fav_wr, unf_wr, w + l + d))
        print(f"  ep{ep:>3}: win_rate={all_wr:.3f} (W={w} L={l} D={d}) "
              f"fav={fav_wr:.3f} unfav={unf_wr:.3f}  [{path.name}]", flush=True)
        del executor, agent

    print("\n=== 勝率曲線 (vs 固定ルール) ===")
    print("  epoch | win_rate | fav(SE可) | unfav | n")
    for ep, wr, fav_wr, unf_wr, n in rows:
        print(f"  {ep:>5} |  {wr:.3f}   |   {fav_wr:.3f}   | {unf_wr:.3f} | {n}")


if __name__ == "__main__":
    main()
