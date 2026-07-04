"""チェックポイントから自己対戦を回し、対面ごとの「モデル生 policy / 教師 target_pi /
selection_pi / 実際の選択頻度」を集計して表示する診断ツール。学習はせず (チェックポイント
を上書きしない)、推論だけで lookahead 教師付き trajectory を集める。

(active の技 × 相手 active 種族) の 4 対面で分類する:
  SW_v_Cl=Shock Wave vs Cloyster (SE/有利)   SW_v_Go=Shock Wave vs Goodra (半減/不利)
  BD_v_Cl=Bulldoze   vs Cloyster (等倍)       BD_v_Go=Bulldoze   vs Goodra (SE/有利)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .agent import Agent
from poke_ai3 import MAX_MOVE_SLOTS

from .diagnostics import (
    SWITCH_MATCHUP_KEYS,
    _legal_move_slots,
    _switch_matchup_key,
)
from .encoding import encode_observations
from .train_loop import get_rust_async_executor_wrapper

# 対面キー → 人間向け説明 (技の有効度メモ付き)。
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
# HP 区分 (ビン境界): hi=(0.66,1.0], mid=(0.33,0.66], lo=[0.0,0.33]。
_HP_BINS = ["hi", "mid", "lo"]


def _hp_bin(frac: float) -> str:
    if frac > 0.66:
        return "hi"
    if frac > 0.33:
        return "mid"
    return "lo"


def _matchup_key(state: dict[str, Any]) -> str | None:
    """(active の技 × 相手 active 種族) の対面キー。交代が合法で active が
    SW/BD のどちらかを撃てるターンだけ対象にする。"""
    return _switch_matchup_key(state)


def collect_items(
    executor: Any,
    agent: Agent,
    num_batches: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    """num_batches 回 trajectory フラッシュを受けるまで推論ループを回し、全 item を平坦化。
    agent.learn は呼ばない (学習・チェックポイント保存をしない)。"""
    items: list[dict[str, Any]] = []
    batches = 0
    while batches < num_batches:
        if executor.trajectories_ready():
            payload = json.loads(executor.recv_trajectories())
            for trajectory in payload.get("vec", []):
                winner = trajectory.get("winner")
                for it in trajectory.get("items", []):
                    # その試合の最終結果を各 item に付ける (盤面 value の実測比較用)。
                    it["_winner"] = winner
                    items.append(it)
            batches += 1
            print(f"  collected batch {batches}/{num_batches} (items so far: {len(items)})")
        elif executor.is_ready():
            agent.infer_step(executor)
        else:
            time.sleep(sleep_seconds)
    return items


def _model_outputs(
    agent: Agent, items: list[dict[str, Any]]
) -> tuple[list[list[float]], list[float]]:
    """モデルの softmax policy と value (盤面勝率予測 0..1) を返す。"""
    encoded = encode_observations(items, agent.device)
    agent.model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=agent.device.type,
            dtype=agent.agent_config.amp_dtype,
            enabled=agent.device.type == "cuda",
        ):
            logits, values = agent.model(encoded)
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        values_cpu = values.float().cpu().tolist()
    return probs, values_cpu


def _realized(item: dict[str, Any]) -> float:
    """その item の手番プレイヤー視点の実際の試合結果 (勝=1.0/負=0.0/引分=0.5)。"""
    winner = item.get("_winner")
    if winner is None:
        return 0.5
    return 1.0 if str(winner) == str(item["player"]) else 0.0


def _new_acc() -> dict[str, Any]:
    return {
        "n": 0,
        "model_switch": 0.0,
        "target_switch": 0.0,
        "selection_switch": 0.0,
        "chosen_switch": 0,
        # lookahead 手別 rollout 勝率: 攻撃 (active 技) と交代 (控え) の平均。
        "attack_wr": 0.0,
        "switch_wr": 0.0,
        # 盤面勝率 (value): モデル予測・教師 target_value・実際の試合結果。
        "model_value": 0.0,
        "target_value": 0.0,
        "realized": 0.0,
        # HP 別 (my_bin, opp_bin): [交代率和, 件数, 攻撃wr和, 交代wr和,
        #                           model_value和, target_value和, realized和]。
        "hp": {(m, o): [0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]
               for m in _HP_BINS for o in _HP_BINS},
    }


def analyze(
    agent: Agent, items: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """対面ごとに model/target/selection の交代確率と実選択頻度、HP 別モデル交代確率を集計。"""
    probs, values = _model_outputs(agent, items)
    acc = {k: _new_acc() for k in SWITCH_MATCHUP_KEYS}
    for item, prob, mval in zip(items, probs, values):
        state = item["state"]
        key = _matchup_key(state)
        if key is None:
            continue
        a = acc[key]
        a["n"] += 1
        model_switch = sum(prob[MAX_MOVE_SLOTS:])
        a["model_switch"] += model_switch
        target_pi = [float(x) for x in item["target_pi"]]
        a["target_switch"] += sum(target_pi[MAX_MOVE_SLOTS:])
        selection_pi = [float(x) for x in item.get("selection_pi", target_pi)]
        a["selection_switch"] += sum(selection_pi[MAX_MOVE_SLOTS:])
        a["chosen_switch"] += int(int(item.get("chosen_action", 0)) >= MAX_MOVE_SLOTS)
        # lookahead 手別勝率: 攻撃 = active 技スロット、交代 = 控えスロットの最大。
        legal_mask = [bool(x) for x in state["legal_action_mask"]]
        win_rates = [float(x) for x in item.get("win_rates", [0.0] * len(prob))]
        move_slots = _legal_move_slots(state)
        active_move = move_slots[0] if len(move_slots) == 1 else None
        attack_wr = win_rates[active_move] if active_move is not None else 0.0
        switch_wrs = [win_rates[i] for i in range(MAX_MOVE_SLOTS, len(win_rates))
                      if i < len(legal_mask) and legal_mask[i]]
        switch_wr = max(switch_wrs) if switch_wrs else 0.0
        a["attack_wr"] += attack_wr
        a["switch_wr"] += switch_wr
        # 盤面勝率: モデル予測 / 教師 / 実結果。
        mv = float(mval) if not isinstance(mval, list) else float(mval[0])
        tval = float(item["target_value"])
        realized = _realized(item)
        a["model_value"] += mv
        a["target_value"] += tval
        a["realized"] += realized
        cell = a["hp"][(
            _hp_bin(float(state["my_exact_hp_frac"])),
            _hp_bin(float(state["opp_quantized_hp_frac"])),
        )]
        cell[0] += model_switch
        cell[1] += 1
        cell[2] += attack_wr
        cell[3] += switch_wr
        cell[4] += mv
        cell[5] += tval
        cell[6] += realized
    return acc


def _rate(num: float, den: int) -> str:
    return "  n/a " if den <= 0 else f"{num / den:.3f}"


def print_report(acc: dict[str, dict[str, Any]]) -> None:
    for key in SWITCH_MATCHUP_KEYS:
        a = acc[key]
        n = a["n"]
        desc = _MATCHUP_DESC.get(key, key)
        print(f"\n=== {key}  {desc}  n={n} ===")
        if n == 0:
            print("  (該当サンプルなし)")
            continue
        # 交代率を出す (攻撃率 = 1 - 交代率)。
        print(f"  model      switch={_rate(a['model_switch'], n)}  attack={_rate(n - a['model_switch'], n)}")
        print(f"  target_pi  switch={_rate(a['target_switch'], n)}  attack={_rate(n - a['target_switch'], n)}")
        print(f"  selection  switch={_rate(a['selection_switch'], n)}  attack={_rate(n - a['selection_switch'], n)}")
        print(f"  chosen     switch={_rate(a['chosen_switch'], n)}  attack={_rate(n - a['chosen_switch'], n)}  (実際の着手頻度)")
        # lookahead 手別勝率: 交代の方が高ければ交代有利。差≈0 なら無差別 (勾配なし)。
        a_wr, s_wr = a["attack_wr"] / n, a["switch_wr"] / n
        print(f"  lookahead win_rate: attack={a_wr:.3f}  switch={s_wr:.3f}  diff(sw-at)={s_wr - a_wr:+.3f}")
        print("  HP別 [行=自分HP,列=相手HP] = 交代率 / win_rate差(sw-at) (n):")
        print("            " + "".join(f"opp_{b:<14}" for b in _HP_BINS))
        for mb in _HP_BINS:
            cells = []
            for ob in _HP_BINS:
                s, c, awr, swr = a["hp"][(mb, ob)][:4]
                if c <= 0:
                    cells.append("  n/a".ljust(18))
                else:
                    diff = (swr - awr) / c
                    cells.append(f"{s / c:.2f}/{diff:+.2f}({c})".ljust(18))
            print(f"    my_{mb:<5}" + "".join(cells))
        # 盤面勝率 (value) の学習度: model 予測 vs 教師 vs 実結果。
        print(f"  盤面勝率 value: model={a['model_value'] / n:.3f}  "
              f"target={a['target_value'] / n:.3f}  realized={a['realized'] / n:.3f}")
        print("  HP別 value [model/target/realized (n)]:")
        print("            " + "".join(f"opp_{b:<16}" for b in _HP_BINS))
        for mb in _HP_BINS:
            cells = []
            for ob in _HP_BINS:
                c = a["hp"][(mb, ob)][1]
                if c <= 0:
                    cells.append("  n/a".ljust(20))
                else:
                    _, _, _, _, mv, tv, rz = a["hp"][(mb, ob)]
                    cells.append(f"{mv / c:.2f}/{tv / c:.2f}/{rz / c:.2f}({c})".ljust(20))
            print(f"    my_{mb:<5}" + "".join(cells))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--num-games", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4,
                        help="集計に使う trajectory フラッシュ回数。")
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
    )
    agent = Agent(device=args.device, checkpoint_path=args.checkpoint_path)
    print(f"checkpoint={args.checkpoint_path} training_step={agent.training_step} "
          f"stage={args.stage} num_games={args.num_games} num_batches={args.num_batches}")
    items = collect_items(executor, agent, args.num_batches, args.sleep_seconds)
    print(f"\ntotal items collected: {len(items)}")
    print_report(analyze(agent, items))


if __name__ == "__main__":
    main()
