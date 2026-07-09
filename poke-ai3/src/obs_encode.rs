//! 観測バッチ → モデル入力テンソルの最終形 (フラット配列群) への Rust エンコード。
//!
//! Python 側 `encoding.py` の `encode_observations` と同じレイアウトを構築する。
//! pyo3 には依存せず、`poke-ai3-python` が rust-numpy で numpy 配列に包んで
//! ゼロコピー返却する。レイアウト定数 (`MAX_MOVE_SLOTS`/`NUM_BENCH`/`ACTION_DIM`) は
//! 従来どおり Rust が唯一の正。
//!
//! empty 観測 (state=None、ウィンドウ穴埋め用) はテンソルに載せず、ルーティング
//! 情報 (`empty_*`) だけを別配列で返す。Python は推論結果と一緒にそのまま
//! エコーバックし、ack タイミングを GPU ラウンドと同期させる (Rust 側で即 ack
//! すると empty が閾値を圧迫するため)。

use crate::root_task::PlayerObservation;
use poke_env_rust::observation::{
    ACTION_DIM, BenchSlot, MAX_MOVE_SLOTS, NUM_BENCH, Player, StateForPlayer,
};

/// バッチ化済み観測。`rows` を B として、多次元配列は行優先のフラット Vec で持つ。
/// 形状はフィールドコメントのとおり (encoding.py の `EncodedObservations` と同一)。
#[derive(Debug, Default, PartialEq)]
pub struct EncodedBatch {
    /// 実観測 (state=Some) の行数 B。
    pub rows: usize,
    // ルーティング (実観測、行順はテンソルと一致)
    pub game_id: Vec<i64>,
    /// (game_id スロットの何ゲーム目か。敵混合の per-game 割り当て鍵。実観測のみ。)
    pub game_index: Vec<i64>,
    pub player: Vec<i64>,
    pub request_id: Vec<i64>,
    // gid 系 (i64)
    pub my_species: Vec<i64>,
    pub opp_species: Vec<i64>,
    /// (B, MAX_MOVE_SLOTS)
    pub move_gids: Vec<i64>,
    /// (B, MAX_MOVE_SLOTS) 相手 active の技 (神視点)
    pub opp_move_gids: Vec<i64>,
    /// (B, NUM_BENCH)
    pub bench_species: Vec<i64>,
    /// (B, NUM_BENCH, MAX_MOVE_SLOTS)
    pub bench_move_gids: Vec<i64>,
    /// (B, NUM_BENCH) 相手控え
    pub opp_bench_species: Vec<i64>,
    /// (B, NUM_BENCH, MAX_MOVE_SLOTS)
    pub opp_bench_move_gids: Vec<i64>,
    // 特徴系 (f32)
    pub my_hp: Vec<f32>,
    pub opp_hp: Vec<f32>,
    /// (B, MAX_MOVE_SLOTS)
    pub move_present: Vec<f32>,
    /// (B, MAX_MOVE_SLOTS)
    pub move_legal: Vec<f32>,
    /// (B, NUM_BENCH)
    pub bench_hp: Vec<f32>,
    /// (B, NUM_BENCH)
    pub bench_present: Vec<f32>,
    /// (B, NUM_BENCH, MAX_MOVE_SLOTS)
    pub bench_move_present: Vec<f32>,
    /// (B, MAX_MOVE_SLOTS)
    pub opp_move_present: Vec<f32>,
    /// (B, NUM_BENCH)
    pub opp_bench_hp: Vec<f32>,
    /// (B, NUM_BENCH)
    pub opp_bench_present: Vec<f32>,
    /// (B, NUM_BENCH, MAX_MOVE_SLOTS)
    pub opp_bench_move_present: Vec<f32>,
    /// (B, NUM_BENCH)
    pub switch_legal: Vec<f32>,
    /// (B, ACTION_DIM)
    pub legal_action_mask: Vec<bool>,
    // empty 観測 (state=None) のルーティング。Python はそのままエコーバックする。
    pub empty_game_id: Vec<i64>,
    pub empty_player: Vec<i64>,
    pub empty_request_id: Vec<i64>,
}

/// gid 列を固定幅 `width` へ。`out_gids` には 0 埋め・切り詰め済み gid、
/// `out_present` には 1.0/0.0 の present フラグを push する。
fn push_padded_gids(gids: &[u16], width: usize, out_gids: &mut Vec<i64>, out_present: &mut Vec<f32>) {
    for i in 0..width {
        match gids.get(i) {
            Some(&g) => {
                out_gids.push(g as i64);
                out_present.push(1.0);
            }
            None => {
                out_gids.push(0);
                out_present.push(0.0);
            }
        }
    }
}

/// 控え列 (自軍/相手共通フォーマット) を固定長 NUM_BENCH でフラット配列群へ push する。
/// 空き枠は species=0 / hp=0 / present=0 / 技 0 埋め。
fn push_bench_slots(
    bench: &[Option<BenchSlot>],
    species: &mut Vec<i64>,
    hp: &mut Vec<f32>,
    present: &mut Vec<f32>,
    move_gids: &mut Vec<i64>,
    move_present: &mut Vec<f32>,
) {
    for i in 0..NUM_BENCH {
        match bench.get(i).and_then(|s| s.as_ref()) {
            Some(slot) => {
                species.push(slot.species_gid as i64);
                hp.push(slot.hp_frac);
                present.push(1.0);
                push_padded_gids(&slot.move_gids, MAX_MOVE_SLOTS, move_gids, move_present);
            }
            None => {
                species.push(0);
                hp.push(0.0);
                present.push(0.0);
                move_gids.extend(std::iter::repeat_n(0, MAX_MOVE_SLOTS));
                move_present.extend(std::iter::repeat_n(0.0, MAX_MOVE_SLOTS));
            }
        }
    }
}

/// 観測バッチをフラット配列群へエンコードする。
pub fn encode_batch(observations: &[PlayerObservation]) -> EncodedBatch {
    let mut b = EncodedBatch::default();
    for obs in observations {
        let Some(state) = &obs.state else {
            b.empty_game_id.push(obs.game_id as i64);
            b.empty_player.push(obs.player.index() as i64);
            b.empty_request_id.push(obs.request_id as i64);
            continue;
        };
        b.rows += 1;
        b.game_id.push(obs.game_id as i64);
        b.game_index.push(obs.game_index as i64);
        b.player.push(obs.player.index() as i64);
        b.request_id.push(obs.request_id as i64);

        b.my_species.push(state.my_species_gid as i64);
        b.opp_species.push(state.opp_species_gid as i64);
        b.my_hp.push(state.my_exact_hp_frac);
        b.opp_hp.push(state.opp_quantized_hp_frac);

        // legal mask は ACTION_DIM へ 0 埋め・切り詰め (encoding.py と同じ防御)。
        let legal =
            |i: usize| -> bool { state.legal_action_mask.get(i).copied().unwrap_or(false) };
        for i in 0..ACTION_DIM {
            b.legal_action_mask.push(legal(i));
        }
        push_padded_gids(&state.my_move_gids, MAX_MOVE_SLOTS, &mut b.move_gids, &mut b.move_present);
        push_padded_gids(
            &state.opp_move_gids,
            MAX_MOVE_SLOTS,
            &mut b.opp_move_gids,
            &mut b.opp_move_present,
        );
        for i in 0..MAX_MOVE_SLOTS {
            b.move_legal.push(if legal(i) { 1.0 } else { 0.0 });
        }

        push_bench_slots(
            &state.my_bench,
            &mut b.bench_species,
            &mut b.bench_hp,
            &mut b.bench_present,
            &mut b.bench_move_gids,
            &mut b.bench_move_present,
        );
        // 交代の合法性は自軍控えのみに存在する (相手側にはない)。
        for i in 0..NUM_BENCH {
            let occupied = state.my_bench.get(i).is_some_and(|s| s.is_some());
            b.switch_legal
                .push(if occupied && legal(MAX_MOVE_SLOTS + i) { 1.0 } else { 0.0 });
        }
        push_bench_slots(
            &state.opp_bench,
            &mut b.opp_bench_species,
            &mut b.opp_bench_hp,
            &mut b.opp_bench_present,
            &mut b.opp_bench_move_gids,
            &mut b.opp_bench_move_present,
        );
    }
    b
}

/// 観測 state 列 (ルーティング情報なし) をエンコードする。play_gui / 学習バッチ構築など
/// 推論ルーティングを伴わない経路用。`game_id`/`request_id` には行番号、`player` には P1 を
/// ダミーとして詰める (テンソル列には影響しない)。
pub fn encode_states(states: &[StateForPlayer]) -> EncodedBatch {
    let observations: Vec<PlayerObservation> = states
        .iter()
        .enumerate()
        .map(|(i, state)| PlayerObservation {
            game_id: i,
            game_index: 0,
            player: Player::P1,
            request_id: i as u64,
            state: Some(state.clone()),
        })
        .collect();
    encode_batch(&observations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_env_rust::observation::{BenchSlot, Player, StateForPlayer};

    fn obs(game_id: usize, request_id: u64, state: Option<StateForPlayer>) -> PlayerObservation {
        PlayerObservation {
            game_id,
            game_index: 0,
            player: Player::P2,
            request_id,
            state,
        }
    }

    fn sample_state() -> StateForPlayer {
        let mut mask = vec![false; ACTION_DIM];
        mask[0] = true;
        mask[MAX_MOVE_SLOTS] = true;
        StateForPlayer {
            my_species_gid: 7,
            opp_species_gid: 11,
            my_exact_hp_frac: 0.75,
            opp_quantized_hp_frac: 0.5,
            my_move_gids: vec![3, 4],
            opp_move_gids: vec![6],
            my_bench: {
                let mut bench = vec![None; NUM_BENCH];
                bench[0] = Some(BenchSlot {
                    species_gid: 9,
                    hp_frac: 0.25,
                    move_gids: vec![5],
                });
                bench
            },
            opp_bench: {
                let mut bench = vec![None; NUM_BENCH];
                bench[0] = Some(BenchSlot {
                    species_gid: 13,
                    hp_frac: 0.5,
                    move_gids: vec![8, 9],
                });
                bench
            },
            legal_action_mask: mask,
        }
    }

    #[test]
    fn encodes_real_and_empty_rows() {
        let batch = encode_batch(&[
            obs(2, 10, Some(sample_state())),
            obs(3, 20, None),
        ]);
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.game_id, vec![2]);
        assert_eq!(batch.player, vec![1]);
        assert_eq!(batch.request_id, vec![10]);
        assert_eq!(batch.empty_game_id, vec![3]);
        assert_eq!(batch.empty_request_id, vec![20]);

        assert_eq!(batch.my_species, vec![7]);
        assert_eq!(batch.opp_species, vec![11]);
        assert_eq!(batch.my_hp, vec![0.75]);
        assert_eq!(batch.opp_hp, vec![0.5]);

        // 技スロット: 2 技 + 0 埋め。
        assert_eq!(batch.move_gids.len(), MAX_MOVE_SLOTS);
        assert_eq!(&batch.move_gids[..2], &[3, 4]);
        assert!(batch.move_gids[2..].iter().all(|&g| g == 0));
        assert_eq!(&batch.move_present[..2], &[1.0, 1.0]);
        assert!(batch.move_present[2..].iter().all(|&p| p == 0.0));
        assert_eq!(batch.move_legal[0], 1.0);
        assert!(batch.move_legal[1..].iter().all(|&l| l == 0.0));

        // 控え: slot0 のみ実在。
        assert_eq!(batch.bench_species[0], 9);
        assert!(batch.bench_species[1..].iter().all(|&g| g == 0));
        assert_eq!(batch.bench_hp[0], 0.25);
        assert_eq!(batch.bench_present[0], 1.0);
        assert!(batch.bench_present[1..].iter().all(|&p| p == 0.0));
        assert_eq!(batch.bench_move_gids.len(), NUM_BENCH * MAX_MOVE_SLOTS);
        assert_eq!(batch.bench_move_gids[0], 5);
        assert_eq!(batch.bench_move_present[0], 1.0);
        assert_eq!(batch.switch_legal[0], 1.0);
        assert!(batch.switch_legal[1..].iter().all(|&l| l == 0.0));

        assert_eq!(batch.legal_action_mask.len(), ACTION_DIM);
        assert!(batch.legal_action_mask[0]);
        assert!(batch.legal_action_mask[MAX_MOVE_SLOTS]);

        // 相手 active の技: 1 技 + 0 埋め。
        assert_eq!(batch.opp_move_gids.len(), MAX_MOVE_SLOTS);
        assert_eq!(batch.opp_move_gids[0], 6);
        assert!(batch.opp_move_gids[1..].iter().all(|&g| g == 0));
        assert_eq!(batch.opp_move_present[0], 1.0);
        assert!(batch.opp_move_present[1..].iter().all(|&p| p == 0.0));

        // 相手控え: slot0 のみ実在。
        assert_eq!(batch.opp_bench_species[0], 13);
        assert!(batch.opp_bench_species[1..].iter().all(|&g| g == 0));
        assert_eq!(batch.opp_bench_hp[0], 0.5);
        assert_eq!(batch.opp_bench_present[0], 1.0);
        assert!(batch.opp_bench_present[1..].iter().all(|&p| p == 0.0));
        assert_eq!(batch.opp_bench_move_gids.len(), NUM_BENCH * MAX_MOVE_SLOTS);
        assert_eq!(&batch.opp_bench_move_gids[..2], &[8, 9]);
        assert_eq!(&batch.opp_bench_move_present[..2], &[1.0, 1.0]);
    }

    #[test]
    fn encode_states_matches_encode_batch_tensors() {
        let state = sample_state();
        let via_states = encode_states(&[state.clone()]);
        let via_batch = encode_batch(&[obs(0, 0, Some(state))]);
        // ルーティング (player ダミー) 以外の全テンソル列が一致する。
        assert_eq!(via_states.rows, via_batch.rows);
        assert_eq!(via_states.my_species, via_batch.my_species);
        assert_eq!(via_states.opp_species, via_batch.opp_species);
        assert_eq!(via_states.my_hp, via_batch.my_hp);
        assert_eq!(via_states.opp_hp, via_batch.opp_hp);
        assert_eq!(via_states.move_gids, via_batch.move_gids);
        assert_eq!(via_states.move_present, via_batch.move_present);
        assert_eq!(via_states.move_legal, via_batch.move_legal);
        assert_eq!(via_states.bench_species, via_batch.bench_species);
        assert_eq!(via_states.bench_hp, via_batch.bench_hp);
        assert_eq!(via_states.bench_present, via_batch.bench_present);
        assert_eq!(via_states.bench_move_gids, via_batch.bench_move_gids);
        assert_eq!(via_states.bench_move_present, via_batch.bench_move_present);
        assert_eq!(via_states.switch_legal, via_batch.switch_legal);
        assert_eq!(via_states.opp_move_gids, via_batch.opp_move_gids);
        assert_eq!(via_states.opp_move_present, via_batch.opp_move_present);
        assert_eq!(via_states.opp_bench_species, via_batch.opp_bench_species);
        assert_eq!(via_states.opp_bench_hp, via_batch.opp_bench_hp);
        assert_eq!(via_states.opp_bench_present, via_batch.opp_bench_present);
        assert_eq!(via_states.opp_bench_move_gids, via_batch.opp_bench_move_gids);
        assert_eq!(via_states.opp_bench_move_present, via_batch.opp_bench_move_present);
        assert_eq!(via_states.legal_action_mask, via_batch.legal_action_mask);
    }

    #[test]
    fn short_legal_mask_is_zero_padded() {
        let mut state = sample_state();
        state.legal_action_mask = vec![true]; // 短い mask でも panic しない
        let batch = encode_batch(&[obs(0, 1, Some(state))]);
        assert!(batch.legal_action_mask[0]);
        assert!(batch.legal_action_mask[1..].iter().all(|&m| !m));
    }
}
