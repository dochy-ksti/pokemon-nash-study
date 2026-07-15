"""2 ステージの厳密 Nash 方策同士を対戦させ、到達局面上の行動頻度を比較する。

既定は Stage 3b vs 3d。`--stages 3c 3e` で対称版の比較に切り替える。方策の形式
(1技=P(交代) か 3技=4行動完全方策か) は npz の policy 配列長から導出するので、
ステージ固有の分岐を持たない。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from poke_ai3._native import HumanGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "poke-ai3" / "nash_geo"
ACTION_NAMES = ("Crunch", "Dark Pulse", "coverage", "Switch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages", nargs=2, metavar=("A", "B"), default=["3b", "3d"],
        help="比較する2ステージ (既定: 3b 3d)",
    )
    parser.add_argument("--games-per-config", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def bucket(frac: float, h: int = 26) -> int:
    if frac <= 0.0:
        return 0
    return max(1, min(h - 1, math.floor(frac * (h - 1) + 0.5)))


def dense_index(team: int, own: dict, opp: dict, species: tuple[int, int], h: int = 26) -> int:
    def side(obs: dict) -> tuple[int, int, int]:
        active = 0 if obs["my_species_gid"] == species[0] else 1
        hp = [0, 0]
        hp[active] = bucket(obs["my_exact_hp_frac"], h)
        bench = obs["my_bench"][0]
        if bench is not None:
            bi = 0 if bench["species_gid"] == species[0] else 1
            hp[bi] = bucket(bench["hp_frac"], h)
        return active, hp[0], hp[1]

    aa, ac, ag = side(own)
    oa, oc, og = side(opp)
    k = team
    for radix, value in ((2, aa), (h, ac), (h, ag), (2, oa), (h, oc), (h, og)):
        k = k * radix + value
    return k


def sample_action(rng: random.Random, probs: np.ndarray, legal: list[bool], full: bool) -> int:
    # env の行動 index は習得技スロット相対。技1本なら (技, 交代)、3本なら
    # (Crunch, Dark Pulse, coverage, 交代) で、交代は常に MAX_MOVE_SLOTS=4。
    env_slots = (0, 1, 2, 4) if full else (0, 4)
    p = np.asarray(probs, dtype=np.float64).copy()
    for i, slot in enumerate(env_slots):
        if not legal[slot]:
            p[i] = 0.0
    total = float(p.sum())
    if total <= 0.0:
        return next(slot for slot in env_slots if legal[slot])
    x = rng.random() * total
    acc = 0.0
    for slot, prob in zip(env_slots, p, strict=True):
        acc += float(prob)
        if x <= acc:
            return slot
    return env_slots[-1]


def tiebreak(obs1: dict, obs2: dict) -> float:
    def hp(obs: dict) -> tuple[int, float]:
        vals = [obs["my_exact_hp_frac"]]
        vals.extend(x["hp_frac"] for x in obs["my_bench"] if x is not None)
        return sum(x > 0 for x in vals), sum(vals)

    a, b = hp(obs1), hp(obs2)
    if a[0] != b[0]:
        return 1.0 if a[0] > b[0] else 0.0
    if abs(a[1] - b[1]) < 1e-9:
        return 0.5
    return 1.0 if a[1] > b[1] else 0.0


def run_stage(stage: str, games_per_config: int, seed: int) -> dict:
    path = DATA / f"nash_geo_h26_{stage}.npz"
    if stage == "3b" and not path.exists():
        path = DATA / "nash_geo_h26.npz"
    data = np.load(path)
    raw_policy = data["policy"]
    # 4行動完全方策か P(交代) 単値かを配列長から導出する (ステージ名に依存しない)。
    full = raw_policy.size == data["value"].size * 4
    if full:
        raw4 = raw_policy.reshape(-1, 4)
        valid = raw4[:, 0] != 0xFFFF
        table = raw4.astype(np.float64) / 1000.0
    else:
        valid = raw_policy != 0xFFFF
        ps = raw_policy.astype(np.float64) / 1000.0
        table = np.stack((1.0 - ps, ps), axis=1)

    mean_policy = table[valid].mean(axis=0)
    full_state = {
        "valid_states": int(valid.sum()),
        "mean_policy": {
            name: float(prob)
            for name, prob in zip(
                ACTION_NAMES if full else ("coverage", "Switch"),
                mean_policy,
                strict=True,
            )
        },
        "mixed_states": int(((table[valid] > 0.02).sum(axis=1) >= 2).sum()),
        "exploit_mean": float(data["exploit_mean"]),
        "exploit_max": float(data["exploit_max"]),
    }

    probe0 = json.loads(HumanGame("team1", 0, 0, stage).observation(1))
    probe1 = json.loads(HumanGame("team1", 1, 0, stage).observation(1))
    species = (probe0["my_species_gid"], probe1["my_species_gid"])
    initial = []
    for team in range(2):
        for own_lead in range(2):
            for opp_lead in range(2):
                game = HumanGame(f"team{team + 1}", own_lead, opp_lead, stage)
                own = json.loads(game.observation(1))
                opp = json.loads(game.observation(2))
                k = dense_index(team, own, opp, species)
                initial.append({
                    "team": team + 1,
                    "own_lead": own_lead,
                    "opp_lead": opp_lead,
                    "value": float(data["value"][k]) / 1000.0,
                    "policy": table[k].tolist(),
                })
    rng = random.Random(seed)
    actions: Counter[str] = Counter()
    by_context: dict[str, Counter[str]] = defaultdict(Counter)
    state_visits: Counter[int] = Counter()
    wins = 0.0
    turns = 0
    games = 0
    natural_ends = 0
    geometric_ends = 0

    for team in range(2):
        for lead1 in range(2):
            for lead2 in range(2):
                for rep in range(games_per_config):
                    game_seed = seed ^ (team << 40) ^ (lead1 << 36) ^ (lead2 << 32) ^ rep
                    game = HumanGame(
                        f"team{team + 1}", lead1, lead2, stage, game_seed, 0
                    )
                    while True:
                        o1 = json.loads(game.observation(1))
                        o2 = json.loads(game.observation(2))
                        k1 = dense_index(team, o1, o2, species)
                        k2 = dense_index(1 - team, o2, o1, species)
                        p1, p2 = table[k1], table[k2]
                        a1 = sample_action(rng, p1, o1["legal_action_mask"], full)
                        a2 = sample_action(rng, p2, o2["legal_action_mask"], full)
                        active1 = 0 if o1["my_species_gid"] == species[0] else 1
                        active2 = 0 if o2["my_species_gid"] == species[0] else 1
                        for side_team, own, opp, action, k in (
                            (team, active1, active2, a1, k1),
                            (1 - team, active2, active1, a2, k2),
                        ):
                            if full:
                                name = ACTION_NAMES[3 if action == 4 else action]
                            else:
                                name = "Switch" if action == 4 else "coverage"
                            favorable = (side_team == 0 and own == opp) or (
                                side_team == 1 and own != opp
                            )
                            context = "coverage-SE" if favorable else "coverage-not-SE"
                            actions[name] += 1
                            by_context[context][name] += 1
                            state_visits[k] += 1
                        result = json.loads(game.step_random(a1, a2, rng.random() < 0.5))
                        turns += 1
                        if result["done"]:
                            natural_ends += 1
                            wins += (
                                1.0 if result["winner"] == 1
                                else 0.0 if result["winner"] == 2 else 0.5
                            )
                            break
                        if rng.random() < 0.01:
                            geometric_ends += 1
                            wins += tiebreak(
                                json.loads(game.observation(1)),
                                json.loads(game.observation(2)),
                            )
                            break
                    games += 1

    def fractions(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values())
        return {name: counter[name] / total for name in ACTION_NAMES if counter[name]}

    return {
        "stage": stage,
        "full_state": full_state,
        "initial_states": initial,
        "games": games,
        "win_rate_p1": wins / games,
        "mean_turns": turns / games,
        "natural_end_fraction": natural_ends / games,
        "geometric_end_fraction": geometric_ends / games,
        "actions": dict(actions),
        "action_fractions": fractions(actions),
        "context_action_fractions": {key: fractions(value) for key, value in by_context.items()},
        "top_visited_states": state_visits.most_common(20),
    }


def main() -> None:
    args = parse_args()
    result = {
        "games_per_config": args.games_per_config,
        "seed": args.seed,
        "stages": [
            run_stage(stage, args.games_per_config, args.seed + i)
            for i, stage in enumerate(args.stages)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
