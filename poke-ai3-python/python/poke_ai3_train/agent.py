from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models.modernbert_abs import ModernBertAbsConfig, require_flash_attention
from poke_ai3 import ACTION_DIM

from .encoding import encoded_from_arrays, packed_views
from .infer_graph import GraphedInferModel
from .model import PolicyValueModel, action_distribution, default_model_config
from .supervised import SupervisedConfig, SupervisedStats, build_examples, train_supervised


@dataclass(frozen=True)
class AgentConfig:
    learning_rate: float = 1e-4
    amp_dtype: torch.dtype = torch.bfloat16


class Agent:
    """学習エージェント (lookahead 学習版)。
    - 自己対戦の両側 (P1/P2) を同一 policy/value net で推論する。
    - 学習は lookahead が作った (training_pi, value) 教師への教師あり学習
      (cross-entropy + MSE)。"""

    def __init__(
        self,
        device: str | None = None,
        checkpoint_path: Path | None = None,
        agent_config: AgentConfig | None = None,
        model_config: ModernBertAbsConfig | None = None,
        supervised_config: SupervisedConfig | None = None,
        infer_graph: bool = True,
        learning_rate: float | None = None,
        infer_max_batch_size: int | None = None,
    ) -> None:
        if agent_config is None:
            agent_config = (
                AgentConfig(learning_rate=learning_rate)
                if learning_rate is not None
                else AgentConfig()
            )
        self.agent_config = agent_config
        self.supervised_config = supervised_config or SupervisedConfig()
        self.device = torch.device(device or "cuda")
        require_flash_attention()
        self.checkpoint_path = checkpoint_path
        # model_config 未指定なら checkpoint に保存された設定を優先する。これにより
        # hidden_size 等が異なる checkpoint を default_model_config と取り違えて
        # load_state_dict が shape mismatch で落ちるのを防ぐ (eval で必須)。
        if model_config is None:
            model_config = self._model_config_from_checkpoint()
        self.model = PolicyValueModel(model_config or default_model_config()).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.agent_config.learning_rate,
        )
        # CUDA Graphs: 推論の forward+softmax (~136 カーネル) を autocast の
        # キャストごと 1 replay にまとめ、メインスレッドのカーネル発行 CPU コストを
        # 消す。fp32 学習モデルをそのままキャプチャする (AdamW の更新は in-place
        # なので replay に反映され、重み同期は不要)。
        self.graphed_infer: GraphedInferModel | None = None
        if infer_graph and self.device.type == "cuda":
            # CLI の --max-batch-size に合わせてグラフ上限を設定する。None のときは
            # GraphedInferModel 既定 (4096) に委ねる。
            graph_kwargs: dict[str, Any] = {}
            if infer_max_batch_size is not None:
                graph_kwargs["max_batch_size"] = infer_max_batch_size
            self.graphed_infer = GraphedInferModel(
                self.model, self.device,
                autocast_dtype=self.agent_config.amp_dtype,
                **graph_kwargs,
            )
        self.training_step = 0
        if self.checkpoint_path is not None:
            self.load_checkpoint_if_present()
        # CLI で learning_rate を明示した場合は checkpoint 由来の optimizer lr を上書きする
        # (lr スイープ用)。未指定 (None) のときは checkpoint 値を尊重する。
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

    def infer_step(self, executor: Any) -> None:
        """executor から観測バッチ (numpy dict) を受け取り、推論結果を返送する。

        empty 観測のルーティング配列はそのままエコーバックし、ack タイミングを
        GPU ラウンドと同期させる (Rust 側で即 ack すると empty がバッチ閾値を
        圧迫するため)。"""
        obs = executor.recv_observations()
        policy, value = self.infer_encoded(obs)
        executor.send_inference(
            obs["game_id"],
            obs["player"],
            obs["request_id"],
            policy.ravel(),
            value,
            obs["empty_game_id"],
            obs["empty_player"],
            obs["empty_request_id"],
        )

    def infer_encoded(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """numpy dict 観測 (Rust エンコード済み) → (policy (B, ACTION_DIM), value (B,))。

        学習ホットパス (recv_observations) はパック形式 (packed_i64/packed_f32/
        legal_action_mask)、診断系 (encode_observation_states) はフィールド別 dict。"""
        packed = "packed_i64" in obs
        batch_size = int(
            obs["legal_action_mask"].shape[0] if packed else obs["my_species"].shape[0]
        )
        if batch_size == 0:
            return (
                np.zeros((0, ACTION_DIM), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        with torch.no_grad():
            if packed and self.graphed_infer is not None:
                probs, values = self.graphed_infer.infer_packed(obs)
            else:
                if packed:
                    encoded = packed_views(
                        torch.from_numpy(obs["packed_i64"]).to(self.device),
                        torch.from_numpy(obs["packed_f32"]).to(self.device),
                        torch.from_numpy(obs["legal_action_mask"]).to(self.device),
                    )
                else:
                    encoded = encoded_from_arrays(obs, self.device)
                self.model.eval()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.agent_config.amp_dtype,
                    enabled=self.device.type == "cuda",
                ):
                    logits, values = self.model(encoded)
                probs = action_distribution(logits.float()).probs
                values = values.float()
        policy = np.ascontiguousarray(probs.cpu().numpy(), dtype=np.float32)
        value = np.ascontiguousarray(values.cpu().numpy(), dtype=np.float32)
        return policy, value

    def learn(self, trajectories: dict[str, Any]) -> SupervisedStats | None:
        (examples, win_rate, matchup_win_rates,
         enemy_win_rate, enemy_games, enemy_win_by_game) = build_examples(trajectories)
        stats = train_supervised(
            self.model,
            self.optimizer,
            examples,
            win_rate,
            self.device,
            self.supervised_config,
            self.agent_config.amp_dtype,
            matchup_win_rates=matchup_win_rates,
            enemy_win_rate=enemy_win_rate,
            enemy_games=enemy_games,
            enemy_win_by_game=enemy_win_by_game,
        )
        if stats is not None:
            self.training_step += 1
            self.save_checkpoint(stats)
        return stats

    def save_checkpoint(self, stats: SupervisedStats, path: Path | None = None) -> None:
        """checkpoint を保存する。path 省略時は self.checkpoint_path に上書き保存。
        path を渡すと任意の場所へ保存できる (10ep ごとのスナップショット用)。"""
        target = path or self.checkpoint_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "agent_config": {"learning_rate": self.agent_config.learning_rate},
                "model_config": self.model.config.to_dict(),
                "supervised_config": self.supervised_config.__dict__,
                "stats": stats.__dict__,
            },
            target,
        )

    def _model_config_from_checkpoint(self) -> ModernBertAbsConfig | None:
        """checkpoint に保存された model_config を読み出す (モデル構築前に呼ぶ)。
        存在しなければ None を返し、呼び出し側は default_model_config に委ねる。"""
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return None
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        values = checkpoint.get("model_config")
        if values is None:
            return None
        return ModernBertAbsConfig.from_dict(values)

    def load_checkpoint_if_present(self) -> None:
        if self.checkpoint_path is None:
            return
        if not self.checkpoint_path.exists():
            return
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = int(checkpoint.get("training_step", 0))
