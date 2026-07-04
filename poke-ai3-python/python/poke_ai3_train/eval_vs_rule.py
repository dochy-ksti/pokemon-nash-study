"""学習済み AI (P1, lookahead 探索つき) を固定ルール最善手エージェント (P2) と対戦させ、
勝率を測る評価ツール。自己対戦均衡とは別の固定ベースラインで、学習方策の交代タイミングの
妥当性 (有利対面で攻撃・不利対面で交代できているか) を検証する。

ルール側 (P2) の方策 (Rust `rule_agent::rule_choice`):
  - active が相手にSEを撃てる → そのSE技で攻撃
  - 撃てない → 相手にSEを撃てる相方へ交代

学習せず・チェックポイント非上書き。P1 (学習側) の trajectory だけが流れてくる
(P2 のルール側 trajectory は Rust 側で送出しない)。先発対面 (P1 の初手の active技×相手種族)
別に全体勝率を集計する。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .agent import Agent
from poke_ai3 import MAX_MOVE_SLOTS

from .diagnostics import (
    SWITCH_MATCHUP_KEYS,
    _active_move_label,
    _gids,
    _se_set_for,
    _species_label,
)
from .encoding import set_mask_opp_obs
from .train_loop import get_rust_async_executor_wrapper

_MATCHUP_DESC = {
    # 3a/3b
    "SW_v_Cl": "Shock Wave vs Cloyster (SE/有利)",
    "SW_v_Go": "Shock Wave vs Goodra   (半減/不利)",
    "BD_v_Cl": "Bulldoze   vs Cloyster (等倍)",
    "BD_v_Go": "Bulldoze   vs Goodra   (SE/有利)",
    # 3c (対称対面)
    "FS_v_Cl": "FightSpe60 vs Cloyster (SE/有利)",
    "FS_v_Go": "FightSpe60 vs Goodra   (等倍)",
    "FP_v_Cl": "FairyPhy60 vs Cloyster (等倍)",
    "FP_v_Go": "FairyPhy60 vs Goodra   (SE/有利)",
}
# 先発対面が「P1 active が相手にSEを撃てるか」= 有利/不利の二分。
# active が初手からSEを通せる有利対面 (3a/3b の SW_v_Cl, BD_v_Go と 3c の FS_v_Cl, FP_v_Go)。
_SE_MATCHUPS = {"SW_v_Cl", "BD_v_Go", "FS_v_Cl", "FP_v_Go"}


def _start_matchup(trajectory: dict[str, Any]) -> str | None:
    """trajectory の初手 (turn1, 満タン) から先発対面キーを判定する。
    P1 の active 技 (SW/BD) × 相手 active 種族 (Cl/Go)。"""
    items = trajectory.get("items", [])
    if not items:
        return None
    state = items[0]["state"]
    opp_label = _species_label(int(state["opp_species_gid"]))
    if opp_label is None:
        return None
    move_label = _active_move_label(state)
    if move_label is None:
        return None
    key = f"{move_label}_v_{opp_label}"
    return key if key in SWITCH_MATCHUP_KEYS else None


def _turn1_switched(trajectory: dict[str, Any]) -> bool | None:
    """P1 の初手 (turn1) が交代だったか。chosen_action >= MAX_MOVE_SLOTS なら交代。"""
    items = trajectory.get("items", [])
    if not items:
        return None
    return int(items[0].get("chosen_action", 0)) >= MAX_MOVE_SLOTS


def collect_results(
    executor: Any,
    agent: Agent,
    num_games_target: int,
    sleep_seconds: float,
) -> dict[str, list[int]]:
    """num_games_target 試合ぶんの P1 trajectory を集めるまで推論ループを回し、
    先発対面別に [win, loss, draw] を計上する。学習はしない。"""
    # 各対面 + "ALL" に [win, loss, draw] カウンタ。さらに初手着手別 (攻撃/交代) の
    # [w,l,d] を per-key に持つ ("<key>@atk" / "<key>@sw")。
    tally: dict[str, list[int]] = {k: [0, 0, 0] for k in SWITCH_MATCHUP_KEYS}
    tally["ALL"] = [0, 0, 0]
    for k in SWITCH_MATCHUP_KEYS:
        tally[f"{k}@atk"] = [0, 0, 0]
        tally[f"{k}@sw"] = [0, 0, 0]
    games = 0
    while games < num_games_target:
        if executor.trajectories_ready():
            payload = json.loads(executor.recv_trajectories())
            for trajectory in payload.get("vec", []):
                winner = trajectory.get("winner")
                # P1 (学習側) 視点: 勝=win / 負=loss / 引分(winner=None)=draw。
                idx = 0 if str(winner) == "P1" else (1 if str(winner) == "P2" else 2)
                key = _start_matchup(trajectory)
                tally["ALL"][idx] += 1
                if key is not None:
                    tally[key][idx] += 1
                    switched = _turn1_switched(trajectory)
                    if switched is not None:
                        tally[f"{key}@{'sw' if switched else 'atk'}"][idx] += 1
                games += 1
            print(f"  games so far: {games}/{num_games_target}")
        elif executor.is_ready():
            agent.infer_step(executor)
        else:
            time.sleep(sleep_seconds)
    return tally


def _fmt(counts: list[int]) -> str:
    win, loss, draw = counts
    n = win + loss + draw
    if n == 0:
        return "  n/a (n=0)"
    wr = win / n
    return (f"勝率={wr:.3f}  (W={win} L={loss} D={draw}, n={n})")


def print_report(tally: dict[str, list[int]]) -> None:
    title = tally.pop("_title", "学習AI(P1) vs 固定ルール(P2)")
    print(f"\n=== {title} 勝率 ===")
    print(f"  全体           {_fmt(tally['ALL'])}")
    print("\n  先発対面別 (P1 初手の active技 × 相手種族):")
    for key in SWITCH_MATCHUP_KEYS:
        tag = "有利" if key in _SE_MATCHUPS else "不利/等倍"
        desc = _MATCHUP_DESC.get(key, key)
        print(f"    {key} [{tag}] {desc}")
        print(f"        全体     {_fmt(tally[key])}")
        # 初手着手別の勝率内訳 (テンポ損仮説の検証)。
        print(f"        初手攻撃 {_fmt(tally[f'{key}@atk'])}")
        print(f"        初手交代 {_fmt(tally[f'{key}@sw'])}")
    # 有利 (SE可) / 非有利 (SE不可) の二分サマリ。
    fav = [sum(x) for x in zip(*(tally[k] for k in SWITCH_MATCHUP_KEYS if k in _SE_MATCHUPS))]
    unfav = [sum(x) for x in zip(*(tally[k] for k in SWITCH_MATCHUP_KEYS if k not in _SE_MATCHUPS))]
    print("\n  二分サマリ:")
    print(f"    先発有利(SE可)   {_fmt(fav)}")
    print(f"    先発不利/等倍    {_fmt(unfav)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--num-games", type=int, default=16,
                        help="並列ゲーム数 (Rust 側 executor)。")
    parser.add_argument("--num-eval-games", type=int, default=512,
                        help="集計対象の総試合数 (これだけ P1 trajectory を集める)。")
    parser.add_argument("--both-rule", dest="both_rule",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="P1 も固定ルールにする (rule vs rule の構造ベースライン)。")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--trajectories-threshold", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
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
    parser.add_argument("--mask-opp-obs", dest="mask_opp_obs",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="相手側拡張観測をゼロ化 (--mask-opp-obs で学習したモデルの評価用)。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mask_opp_obs:
        set_mask_opp_obs(True)
        print("mask_opp_obs: enabled")
    # SE 判定の前提 (種族→弱点技集合) が Rust 側 rule_agent と一致することの確認用。
    # 3a/3b は ShockWave/Bulldoze、3c は FightSpe60/FairyPhy60 が SE。
    g = _gids()
    assert _se_set_for(g["Cloyster"]) == frozenset({g["ShockWave"], g["FightSpe60"]})
    assert _se_set_for(g["GoodraHisui"]) == frozenset({g["Bulldoze"], g["FairyPhy60"]})
    assert _se_set_for(g["Goodra"]) == frozenset({g["FairyPhy60"]})
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
        True,            # eval_rule_opponent: P2 を固定ルール方策にする
        args.both_rule,  # eval_rule_p1: P1 も固定ルール (rule vs rule)
    )
    agent = Agent(device=args.device, checkpoint_path=args.checkpoint_path)
    print(f"checkpoint={args.checkpoint_path} training_step={agent.training_step} "
          f"stage={args.stage} num_games={args.num_games} num_eval_games={args.num_eval_games}")
    tally = collect_results(executor, agent, args.num_eval_games, args.sleep_seconds)
    tally["_title"] = "固定ルール(P1) vs 固定ルール(P2)" if args.both_rule \
        else "学習AI(P1) vs 固定ルール(P2)"
    print_report(tally)


if __name__ == "__main__":
    main()
