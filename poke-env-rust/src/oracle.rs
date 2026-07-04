//! lookahead の推論境界。policy/value net への問い合わせを抽象化する。
//!
//! 純粋な Rust 計算 (rollout) から policy/value の推論だけを `PolicyOracle` 経由で
//! 外部 (Python net) に委譲するためのトレイトと結果型を置く。poke-ai3 側の
//! InferenceClient (Python net への batch 推論) が実装する。

use crate::observation::{ACTION_DIM, Player, StateForPlayer};

/// policy/value 推論の結果。`policy` は legal 手に対する softmax 確率 (長さ ACTION_DIM)。
pub struct OracleOut {
    pub policy: [f32; ACTION_DIM],
    pub value: f32,
}

/// rollout 中の各 ply で policy/value を問い合わせるためのオラクル。
pub trait PolicyOracle {
    fn infer(
        &self,
        state: StateForPlayer,
        player: Player,
    ) -> impl std::future::Future<Output = OracleOut> + Send;

    /// empty (穴埋め用ダミー) を 1 件送って ack を待つ。スライディングウィンドウの空き
    /// スロットがこれを繰り返し、各 lookahead が常に `sim_concurrency` 本を root に計上
    /// させることで threshold ゲートのデッドロックを防ぐ。応答内容は使わない。
    fn ack_empty(&self, player: Player) -> impl std::future::Future<Output = ()> + Send;
}
