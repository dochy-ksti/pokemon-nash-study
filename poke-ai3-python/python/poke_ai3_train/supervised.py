from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .diagnostics import (
    _species_label,
    model_diagnostics,
    stage3b_switch_diagnostics,
    stage3b_switch_diagnostics_per_matchup,
)
from .encoding import encode_observations
from .model import action_distribution

# 学習側の表記 (両者同一 net の自己対戦だが、勝率集計は P1 視点で行う)。
LEARNING_PLAYER = "P1"

# マッチアップ (P1視点 my_v_opp) の列挙順。先発はランダムなので 4 カード全部出る。
MATCHUP_KEYS = ["Cl_v_Cl", "Cl_v_Go", "Go_v_Cl", "Go_v_Go"]


@dataclass(frozen=True)
class SupervisedConfig:
    """lookahead が作った (training_pi, value) 教師に対する教師あり学習の設定。"""

    epochs: int = 4
    minibatch_size: int = 64
    value_coef: float = 0.5
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class SupervisedStats:
    examples: int
    policy_loss: float
    value_loss: float
    entropy: float
    raw_logits_std: float
    mean_target_value: float
    # 相手種族別の弱点技選択率 (モデルの argmax 基準) と全体の弱点技選択率。
    # stage3b は混合戦略のナッシュ均衡で「正解の一手」は無いため、これは正解率では
    # なく「greedy argmax がタイプ弱点技を選ぶ率」という行動プローブ。
    vs_cloyster_weakness_rate: float
    vs_goodra_weakness_rate: float
    weakness_move_rate: float
    # Stage3b: 不利対面でのモデル/教師の交代確率と該当サンプル数 (非該当は -1.0 / 0)。
    # 学習成立は model_switch_prob ≈ teacher_switch_prob (混合戦略の忠実な再現)。
    model_switch_prob: float
    teacher_switch_prob: float
    switch_samples: int
    win_rate: float
    # 敵混合学習: 対敵 (凍結した過去 checkpoint) の P1 勝率と敵ゲーム数。
    # 敵なし (K=0 / warmup) では enemy_win_rate=-1.0, enemy_games=0。
    enemy_win_rate: float
    enemy_games: int
    # 敵ゲームの game_id 別 (wins, losses, draws)。呼び出し側が game_id→敵の対応から
    # 敵別勝率を集計するための生データ。
    enemy_win_by_game: dict[int, tuple[int, int, int]]
    # マッチアップ (P1視点 my_v_opp) ごとの (勝率, 試合数)。先発対面で集計 (自己対戦のみ)。
    matchup_win_rates: dict[str, tuple[float, int]]
    # (active技_v_相手種族) ごとの (モデル交代確率, 教師交代確率, 該当数)。攻撃率は 1-交代率。
    # SW_v_Cl=SE / SW_v_Go=半減 / BD_v_Cl=等倍 / BD_v_Go=SE。
    matchup_switch_rates: dict[str, tuple[float, float, int]]


def _matchup_key(state: dict[str, Any]) -> str | None:
    my = _species_label(int(state["my_species_gid"]))
    opp = _species_label(int(state["opp_species_gid"]))
    if my is None or opp is None:
        return None
    return f"{my}_v_{opp}"


def build_examples(
    trajectories: dict[str, Any],
) -> tuple[
    list[dict[str, Any]], float, dict[str, tuple[float, int]],
    float, int, dict[int, tuple[int, int, int]],
]:
    """trajectory を学習サンプルに展開する。

    敵混合学習の役割は Rust が各 trajectory に `enemy_game` タグで付ける:
    - 自己対戦ゲーム (enemy_game=False): P1/P2 両方を学習サンプルに使う。
    - 敵ゲーム (enemy_game=True): P2 は凍結した過去 checkpoint なので学習教師にせず、
      P1 (学習者) の items のみ使う。
    勝率は P1 trajectory のみから集計 (1 ゲーム 2 trajectory の二重計上を避ける) し、
    自己対戦 (~50%) と対敵で分離する。matchup 診断は自己対戦のみで集計する。
    敵ゲームの勝敗は game_id 別 (wins, losses, draws) でも返し、呼び出し側が
    game_id→敵 checkpoint の対応から敵別勝率を出せるようにする。
    敵なし (K=0 / warmup) では全ゲームが自己対戦なので従来挙動と一致する。"""
    examples: list[dict[str, Any]] = []
    wins = 0
    losses = 0
    draws = 0
    matchup_wins: dict[str, int] = {k: 0 for k in MATCHUP_KEYS}
    matchup_total: dict[str, int] = {k: 0 for k in MATCHUP_KEYS}
    enemy_wins = 0
    enemy_losses = 0
    enemy_draws = 0
    enemy_win_by_game: dict[int, tuple[int, int, int]] = {}
    for trajectory in trajectories.get("vec", []):
        items = trajectory.get("items", [])
        is_enemy_game = bool(trajectory.get("enemy_game", False))
        is_p1 = bool(items) and str(items[0]["player"]) == LEARNING_PLAYER
        # 敵ゲームの P2 (凍結敵) は学習教師にしない。
        if not (is_enemy_game and not is_p1):
            examples.extend(items)
        if is_p1:
            winner = trajectory.get("winner")
            won = winner is not None and str(winner) == LEARNING_PLAYER
            if is_enemy_game:
                gid = int(trajectory.get("game_id", -1))
                w, l, d = enemy_win_by_game.get(gid, (0, 0, 0))
                if winner is None:
                    enemy_draws += 1
                    d += 1
                elif won:
                    enemy_wins += 1
                    w += 1
                else:
                    enemy_losses += 1
                    l += 1
                enemy_win_by_game[gid] = (w, l, d)
            else:
                if winner is None:
                    draws += 1
                elif won:
                    wins += 1
                else:
                    losses += 1
                key = _matchup_key(items[0]["state"])
                if key in matchup_total:
                    matchup_total[key] += 1
                    matchup_wins[key] += int(won)
    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0
    enemy_total = enemy_wins + enemy_losses + enemy_draws
    enemy_win_rate = enemy_wins / enemy_total if enemy_total > 0 else -1.0
    matchup_win_rates = {
        k: ((matchup_wins[k] / matchup_total[k]) if matchup_total[k] > 0 else -1.0,
            matchup_total[k])
        for k in MATCHUP_KEYS
    }
    return (examples, win_rate, matchup_win_rates,
            enemy_win_rate, enemy_total, enemy_win_by_game)


def train_supervised(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: list[dict[str, Any]],
    win_rate: float,
    device: torch.device,
    config: SupervisedConfig,
    amp_dtype: torch.dtype = torch.bfloat16,
    matchup_win_rates: dict[str, tuple[float, int]] | None = None,
    enemy_win_rate: float = -1.0,
    enemy_games: int = 0,
    enemy_win_by_game: dict[int, tuple[int, int, int]] | None = None,
) -> SupervisedStats | None:
    if not examples:
        return None
    model.train()

    encoded = encode_observations(examples, device)
    target_pi = torch.tensor(
        [list(item["target_pi"]) for item in examples],
        dtype=torch.float32,
        device=device,
    )
    target_v = torch.tensor(
        [float(item["target_value"]) for item in examples],
        dtype=torch.float32,
        device=device,
    )

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    raw_logits_stds: list[float] = []
    count = len(examples)

    for _ in range(config.epochs):
        permutation = torch.randperm(count, device=device)
        for start in range(0, count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                logits, values = model(encoded[indices])
            logits = logits.float()
            values = values.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            # soft-target cross-entropy (教師は lookahead の training_pi)。
            policy_loss = -(target_pi[indices] * log_probs).sum(dim=-1).mean()
            value_loss = torch.nn.functional.mse_loss(values, target_v[indices])
            dist = action_distribution(logits)
            entropy = dist.entropy().mean()
            raw_logits_std = logits.detach().float().std().item()
            loss = policy_loss + config.value_coef * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            raw_logits_stds.append(raw_logits_std)

    vs_cl, vs_go, weakness = model_diagnostics(model, encoded, examples, device, amp_dtype)
    model_switch_prob, teacher_switch_prob, switch_samples = stage3b_switch_diagnostics(
        model, encoded, examples, device, amp_dtype
    )
    matchup_switch_rates = stage3b_switch_diagnostics_per_matchup(
        model, encoded, examples, device, amp_dtype
    )
    return SupervisedStats(
        examples=count,
        policy_loss=_mean(policy_losses),
        value_loss=_mean(value_losses),
        entropy=_mean(entropies),
        raw_logits_std=_mean(raw_logits_stds),
        mean_target_value=float(target_v.mean().detach().cpu()),
        vs_cloyster_weakness_rate=vs_cl,
        vs_goodra_weakness_rate=vs_go,
        weakness_move_rate=weakness,
        model_switch_prob=model_switch_prob,
        teacher_switch_prob=teacher_switch_prob,
        switch_samples=switch_samples,
        win_rate=win_rate,
        enemy_win_rate=enemy_win_rate,
        enemy_games=enemy_games,
        enemy_win_by_game=enemy_win_by_game or {},
        matchup_win_rates=matchup_win_rates or {},
        matchup_switch_rates=matchup_switch_rates,
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
