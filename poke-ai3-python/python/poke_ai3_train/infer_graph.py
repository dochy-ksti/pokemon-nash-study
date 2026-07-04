"""CUDA Graphs による推論の launch-bound 対策。

forward + softmax (~136 カーネル) を 1 回のグラフ replay にまとめ、メインスレッドの
カーネル発行 CPU コストを消す。入力は Rust が返すパック行列 (packed_i64 / packed_f32 /
legal_action_mask) で、ホスト→静的バッファの H2D コピー 3 回 + replay だけで推論が完了
する。列→フィールドへの分解 (`packed_views`) はキャプチャ時にグラフへ焼き込まれる。

- バッチサイズは 2 のべき乗バケットに切り上げ、バケットごとに遅延キャプチャする。
  パディング行はゼロ (present=0, mask=False)。行はバッチ内で独立なので実データ行に
  影響せず、出力は先頭 B 行だけ返す。
- 重み更新は in-place (AdamW / load_state_dict の copy_) であること。パラメータの
  storage が差し替わるとキャプチャ済みグラフが古い重みを参照し続ける。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from poke_ai3 import ACTION_DIM

from .encoding import PACKED_F32_WIDTH, PACKED_I64_WIDTH, packed_views

_WARMUP_ITERS = 3


class GraphedInferModel:
    """eval 済みモデルの forward+softmax を CUDA Graph 化した呼び出しラッパ。"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_batch_size: int = 4096,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.device = device
        # 実バッチは bucket (2 のべき乗) に切り上げて replay するため、上限も bucket 境界に
        # 切り上げておく。そうしないと「CLI 上限ちょうどのバッチ」が 1 つ上の bucket に
        # 切り上がって max_batch_size を超え、eager にフォールバックしてしまう
        # (例: 上限 341 → バッチ 341 は bucket 512 > 341 で fallback)。
        self.max_batch_size = self._bucket(max_batch_size)
        # fp32 重みモデルを autocast 込みでキャプチャする場合に指定する。
        # キャスト含め全カーネルがグラフに焼き込まれ、replay は fp32 重みを毎回読む
        # ため学習による重み更新 (in-place) は反映される。
        self.autocast_dtype = autocast_dtype
        # bucket -> (graph, 静的入力 (i64, f32, mask), 静的出力 (probs, values))
        self._graphs: dict[int, tuple[Any, ...]] = {}

    def infer_packed(self, obs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """パック numpy dict (B 行) → (probs (B, ACTION_DIM) f32, values (B,) f32)。

        GPU 上のテンソル (静的出力バッファのスライス) を返す。次の呼び出しで上書き
        されるため、呼び出し側はすぐ CPU へ取り出すこと。"""
        batch = int(obs["legal_action_mask"].shape[0])
        bucket = self._bucket(batch)
        if bucket > self.max_batch_size:
            return self._forward_packed(
                torch.from_numpy(obs["packed_i64"]).to(self.device),
                torch.from_numpy(obs["packed_f32"]).to(self.device),
                torch.from_numpy(obs["legal_action_mask"]).to(self.device),
            )
        if bucket not in self._graphs:
            self._graphs[bucket] = self._capture(bucket)
        graph, st_i64, st_f32, st_mask, probs, values = self._graphs[bucket]
        st_i64[:batch].copy_(torch.from_numpy(obs["packed_i64"]), non_blocking=True)
        st_f32[:batch].copy_(torch.from_numpy(obs["packed_f32"]), non_blocking=True)
        st_mask[:batch].copy_(torch.from_numpy(obs["legal_action_mask"]), non_blocking=True)
        graph.replay()
        # 前回 replay の残骸が bucket 内の B 以降に残るが、行独立なので切り捨てるだけでよい。
        return probs[:batch], values[:batch]

    def _forward_packed(
        self, packed_i64: torch.Tensor, packed_f32: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            enc = packed_views(packed_i64, packed_f32, mask)
            if self.autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    logits, values = self.model(enc)
            else:
                logits, values = self.model(enc)
            return torch.softmax(logits.float(), dim=-1), values.float()

    @staticmethod
    def _bucket(batch: int) -> int:
        size = 8
        while size < batch:
            size *= 2
        return size

    def _capture(self, bucket: int) -> tuple[Any, ...]:
        st_i64 = torch.zeros((bucket, PACKED_I64_WIDTH), dtype=torch.long, device=self.device)
        st_f32 = torch.zeros((bucket, PACKED_F32_WIDTH), dtype=torch.float32, device=self.device)
        st_mask = torch.zeros((bucket, ACTION_DIM), dtype=torch.bool, device=self.device)
        # dropout がグラフに焼き込まれないよう eval を強制してからキャプチャする。
        self.model.eval()
        # キャプチャ前のウォームアップは side stream で行う (公式レシピ)。
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_ITERS):
                self._forward_packed(st_i64, st_f32, st_mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            probs, values = self._forward_packed(st_i64, st_f32, st_mask)
        return graph, st_i64, st_f32, st_mask, probs, values
