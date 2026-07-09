//! `EncodedBatch` を Python 転送用のパック形式へ詰め替える。
//!
//! フィールドを行優先で連結した 3 本の行列 (`packed_i64` (B, KI), `packed_f32`
//! (B, KF), `legal_action_mask` (B, ACTION_DIM)) にまとめ、Python 側の H2D 転送を
//! 配列ごと 21 回 → 3 回に減らす。列レイアウトは `I64_LAYOUT` / `F32_LAYOUT` が
//! 唯一の正で、Python へは `packed_layout_json` (JSON) としてエクスポートする。
//!
//! 詰め替えは RootTask (tokio ワーカー) 側で行い、Python メインスレッド (推論
//! ポンプ) にエンコードコストを載せないこと。

use crate::obs_encode::EncodedBatch;
use poke_env_rust::observation::{MAX_MOVE_SLOTS, NUM_BENCH};

pub const I64_LAYOUT: &[(&str, &[usize])] = &[
    ("my_species", &[]),
    ("opp_species", &[]),
    ("move_gids", &[MAX_MOVE_SLOTS]),
    ("opp_move_gids", &[MAX_MOVE_SLOTS]),
    ("bench_species", &[NUM_BENCH]),
    ("bench_move_gids", &[NUM_BENCH, MAX_MOVE_SLOTS]),
    ("opp_bench_species", &[NUM_BENCH]),
    ("opp_bench_move_gids", &[NUM_BENCH, MAX_MOVE_SLOTS]),
];

pub const F32_LAYOUT: &[(&str, &[usize])] = &[
    ("my_hp", &[]),
    ("opp_hp", &[]),
    ("move_present", &[MAX_MOVE_SLOTS]),
    ("move_legal", &[MAX_MOVE_SLOTS]),
    ("opp_move_present", &[MAX_MOVE_SLOTS]),
    ("bench_hp", &[NUM_BENCH]),
    ("bench_present", &[NUM_BENCH]),
    ("switch_legal", &[NUM_BENCH]),
    ("opp_bench_hp", &[NUM_BENCH]),
    ("opp_bench_present", &[NUM_BENCH]),
    ("bench_move_present", &[NUM_BENCH, MAX_MOVE_SLOTS]),
    ("opp_bench_move_present", &[NUM_BENCH, MAX_MOVE_SLOTS]),
];

/// Python 転送用にパック済みの観測バッチ。
pub struct PackedBatch {
    pub rows: usize,
    /// 列数 (i64 側)。
    pub ki: usize,
    /// 列数 (f32 側)。
    pub kf: usize,
    pub packed_i64: Vec<i64>,
    pub packed_f32: Vec<f32>,
    /// (B, ACTION_DIM) 行優先。
    pub legal_action_mask: Vec<bool>,
    pub game_id: Vec<i64>,
    /// game_id スロットの何ゲーム目か (実観測のみ)。敵混合の per-game 割り当て鍵。
    pub game_index: Vec<i64>,
    pub player: Vec<i64>,
    pub request_id: Vec<i64>,
    pub empty_game_id: Vec<i64>,
    pub empty_player: Vec<i64>,
    pub empty_request_id: Vec<i64>,
}

pub fn packed_layout_json() -> String {
    serde_json::json!({
        "i64": I64_LAYOUT.iter().map(|(n, s)| (n, s.to_vec())).collect::<Vec<_>>(),
        "f32": F32_LAYOUT.iter().map(|(n, s)| (n, s.to_vec())).collect::<Vec<_>>(),
    })
    .to_string()
}

pub fn pack_batch(batch: EncodedBatch) -> PackedBatch {
    let rows = batch.rows;
    let (packed_i64, ki) = pack_rows(rows, I64_LAYOUT, |name| i64_field(&batch, name));
    let (packed_f32, kf) = pack_rows(rows, F32_LAYOUT, |name| f32_field(&batch, name));
    PackedBatch {
        rows,
        ki,
        kf,
        packed_i64,
        packed_f32,
        legal_action_mask: batch.legal_action_mask,
        game_id: batch.game_id,
        game_index: batch.game_index,
        player: batch.player,
        request_id: batch.request_id,
        empty_game_id: batch.empty_game_id,
        empty_player: batch.empty_player,
        empty_request_id: batch.empty_request_id,
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn pack_rows<'a, T: Copy + 'a>(
    rows: usize,
    layout: &[(&str, &[usize])],
    field: impl Fn(&str) -> &'a [T],
) -> (Vec<T>, usize) {
    let width: usize = layout.iter().map(|(_, s)| numel(s)).sum();
    let mut packed = Vec::with_capacity(rows * width);
    for row in 0..rows {
        for (name, shape) in layout {
            let n = numel(shape);
            let data = field(name);
            debug_assert_eq!(data.len(), rows * n, "field {name} length mismatch");
            packed.extend_from_slice(&data[row * n..(row + 1) * n]);
        }
    }
    (packed, width)
}

fn i64_field<'a>(batch: &'a EncodedBatch, name: &str) -> &'a [i64] {
    match name {
        "my_species" => &batch.my_species,
        "opp_species" => &batch.opp_species,
        "move_gids" => &batch.move_gids,
        "opp_move_gids" => &batch.opp_move_gids,
        "bench_species" => &batch.bench_species,
        "bench_move_gids" => &batch.bench_move_gids,
        "opp_bench_species" => &batch.opp_bench_species,
        "opp_bench_move_gids" => &batch.opp_bench_move_gids,
        other => unreachable!("unknown i64 field {other}"),
    }
}

fn f32_field<'a>(batch: &'a EncodedBatch, name: &str) -> &'a [f32] {
    match name {
        "my_hp" => &batch.my_hp,
        "opp_hp" => &batch.opp_hp,
        "move_present" => &batch.move_present,
        "move_legal" => &batch.move_legal,
        "opp_move_present" => &batch.opp_move_present,
        "bench_hp" => &batch.bench_hp,
        "bench_present" => &batch.bench_present,
        "switch_legal" => &batch.switch_legal,
        "opp_bench_hp" => &batch.opp_bench_hp,
        "opp_bench_present" => &batch.opp_bench_present,
        "bench_move_present" => &batch.bench_move_present,
        "opp_bench_move_present" => &batch.opp_bench_move_present,
        other => unreachable!("unknown f32 field {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_covers_all_encoded_fields() {
        // EncodedBatch のテンソル系フィールド数 = i64 8 + f32 12 + mask 1。
        assert_eq!(I64_LAYOUT.len(), 8);
        assert_eq!(F32_LAYOUT.len(), 12);
        let ki: usize = I64_LAYOUT.iter().map(|(_, s)| numel(s)).sum();
        let kf: usize = F32_LAYOUT.iter().map(|(_, s)| numel(s)).sum();
        assert!(ki > 0 && kf > 0);
    }
}
