"""複数 checkpoint が「同一局面で同じ評価を返すか」を測る診断ツール。

仮説: 強い checkpoint は既にナッシュ均衡に達しており、独立に学習した強モデル同士は
どの局面でも (重みを取り替えても) ほぼ同じ policy / value を出すはず。これを検証する。

手順:
  1. 局面収集: 先頭 checkpoint で自己対戦を回し (学習なし)、局面集合を集める。
  2. 重み取り替え評価: 集めた同一局面すべてを各 checkpoint の重みで推論し直す。
  3. 一致度集計: checkpoint ペアごとに
       - policy 総変動距離 TV (合法手で正規化した分布の 0.5*L1)
       - value (盤面勝率予測) 絶対差 MAE
     を出す。強同士の一致が強 vs 弱より際立って高ければ「同一均衡へ収束」を支持する。

混合戦略均衡では best response が複数あり policy は一意とは限らないため、より確実な
「同じ評価」の指標は value の一致である (両方を出力する)。
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import torch

from poke_ai3 import ACTION_DIM

from .agent import Agent
from .encoding import encode_observations
from .inspect_matchup import collect_items
from .train_loop import get_rust_async_executor_wrapper


def _masked_policy(probs: list[float], legal: list[bool]) -> list[float]:
    """合法手だけ残して再正規化した分布を返す。合法手が無ければ一様。"""
    masked = [p if lg else 0.0 for p, lg in zip(probs, legal)]
    total = sum(masked)
    if total <= 0.0:
        n = sum(1 for lg in legal if lg)
        if n == 0:
            return [0.0] * len(probs)
        return [1.0 / n if lg else 0.0 for lg in legal]
    return [m / total for m in masked]


def _model_outputs(
    agent: Agent, items: list[dict[str, Any]]
) -> tuple[list[list[float]], list[float]]:
    """その checkpoint の softmax policy (合法手で再正規化) と value を返す。"""
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
    policies: list[list[float]] = []
    for item, prob in zip(items, probs):
        legal = [bool(x) for x in item["state"]["legal_action_mask"]]
        policies.append(_masked_policy(prob[:ACTION_DIM], legal[:ACTION_DIM]))
    values_flat = [float(v[0]) if isinstance(v, list) else float(v) for v in values_cpu]
    return policies, values_flat


def _tv(p: list[float], q: list[float]) -> float:
    """総変動距離 = 0.5 * L1。0 (完全一致) .. 1 (互いに素)。"""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def pairwise_report(
    labels: list[str],
    policies: list[list[list[float]]],
    values: list[list[float]],
) -> None:
    """checkpoint ペアごとの平均 policy TV と value MAE を表示する。"""
    n_ckpt = len(labels)
    n_item = len(policies[0])
    print(f"\n=== ペアごとの一致度 (n_item={n_item}) ===")
    print("  policy_TV: 0=完全一致 .. 1=互いに素 / value_MAE: 盤面勝率予測の平均絶対差")
    for i, j in itertools.combinations(range(n_ckpt), 2):
        tv_sum = sum(_tv(policies[i][k], policies[j][k]) for k in range(n_item))
        val_sum = sum(abs(values[i][k] - values[j][k]) for k in range(n_item))
        print(
            f"  {labels[i]:>20} vs {labels[j]:<20}  "
            f"policy_TV={tv_sum / n_item:.4f}  value_MAE={val_sum / n_item:.4f}"
        )
    # 各局面での「全 checkpoint の value のばらつき (標準偏差)」の平均。
    print("\n=== 局面ごとの value ばらつき ===")
    sd_sum = 0.0
    for k in range(n_item):
        vals = [values[c][k] for c in range(n_ckpt)]
        mean = sum(vals) / n_ckpt
        var = sum((v - mean) ** 2 for v in vals) / n_ckpt
        sd_sum += var**0.5
    print(f"  局面平均 value 標準偏差 = {sd_sum / n_item:.4f} (0 に近いほど全 ckpt が同評価)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, action="append", required=True,
        help="比較する checkpoint。複数回指定する。先頭が局面収集にも使われる。",
    )
    parser.add_argument("--num-games", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4,
                        help="局面収集に使う trajectory フラッシュ回数。")
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
    if len(args.checkpoint) < 2:
        raise SystemExit("--checkpoint は 2 個以上指定してください。")
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
    # 先頭 checkpoint で局面を収集 (学習はしない)。
    collector = Agent(device=args.device, checkpoint_path=args.checkpoint[0])
    print(f"局面収集 checkpoint={args.checkpoint[0]} step={collector.training_step} "
          f"stage={args.stage} num_games={args.num_games} num_batches={args.num_batches}")
    items = collect_items(executor, collector, args.num_batches, args.sleep_seconds)
    print(f"\n収集局面数: {len(items)}")
    if not items:
        raise SystemExit("局面が集まりませんでした。num-batches を増やしてください。")

    labels: list[str] = []
    policies: list[list[list[float]]] = []
    values: list[list[float]] = []
    for path in args.checkpoint:
        agent = collector if path == args.checkpoint[0] else Agent(
            device=args.device, checkpoint_path=path)
        pol, val = _model_outputs(agent, items)
        labels.append(path.stem)
        policies.append(pol)
        values.append(val)
        print(f"  evaluated {path.stem} (step={agent.training_step})")
        if agent is not collector:
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pairwise_report(labels, policies, values)


if __name__ == "__main__":
    main()
