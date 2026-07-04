//! `EncodedBatch` (Rust 側で組み立てた最終テンソル形) を numpy 配列の dict へ包む。
//!
//! フラット Vec は `into_pyarray` でムーブされコピーなしで numpy 配列になる。
//! キー名は Python 側 `encoding.py::EncodedObservations` のフィールド名と一致させる。

use numpy::{IntoPyArray, PyArrayMethods};
use poke_ai3::obs_encode::{EncodedBatch, encode_states};
use poke_ai3::packed::PackedBatch;
use poke_env_rust::observation::{ACTION_DIM, MAX_MOVE_SLOTS, NUM_BENCH, StateForPlayer};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `StateForPlayer` の JSON 配列をエンコードし、numpy 配列の dict で返す。
/// 学習ホットパス (`recv_observations`) 以外の経路 (play_gui / 学習バッチ構築 /
/// 診断ツール) 用の口。キーは `recv_observations` と同じ (ルーティング系はダミー)。
#[pyfunction]
pub fn encode_observation_states<'py>(
    py: Python<'py>,
    states_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let states: Vec<StateForPlayer> = serde_json::from_str(states_json)
        .map_err(|e| PyValueError::new_err(format!("invalid StateForPlayer JSON: {e}")))?;
    encoded_batch_to_dict(py, encode_states(&states))
}

/// 多次元フィールドを reshape しつつ dict 化する。
pub fn encoded_batch_to_dict<'py>(
    py: Python<'py>,
    batch: EncodedBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let b = batch.rows;
    let dict = PyDict::new(py);

    // ルーティング (1 次元)
    dict.set_item("game_id", batch.game_id.into_pyarray(py))?;
    dict.set_item("player", batch.player.into_pyarray(py))?;
    dict.set_item("request_id", batch.request_id.into_pyarray(py))?;
    dict.set_item("empty_game_id", batch.empty_game_id.into_pyarray(py))?;
    dict.set_item("empty_player", batch.empty_player.into_pyarray(py))?;
    dict.set_item("empty_request_id", batch.empty_request_id.into_pyarray(py))?;

    // 1 次元 (B,)
    dict.set_item("my_species", batch.my_species.into_pyarray(py))?;
    dict.set_item("opp_species", batch.opp_species.into_pyarray(py))?;
    dict.set_item("my_hp", batch.my_hp.into_pyarray(py))?;
    dict.set_item("opp_hp", batch.opp_hp.into_pyarray(py))?;

    // 2 次元 (B, MAX_MOVE_SLOTS)
    let moves = [b, MAX_MOVE_SLOTS];
    set_reshaped(&dict, py, "move_gids", batch.move_gids, &moves)?;
    set_reshaped(&dict, py, "move_present", batch.move_present, &moves)?;
    set_reshaped(&dict, py, "move_legal", batch.move_legal, &moves)?;
    set_reshaped(&dict, py, "opp_move_gids", batch.opp_move_gids, &moves)?;
    set_reshaped(&dict, py, "opp_move_present", batch.opp_move_present, &moves)?;

    // 2 次元 (B, NUM_BENCH)
    let bench = [b, NUM_BENCH];
    set_reshaped(&dict, py, "bench_species", batch.bench_species, &bench)?;
    set_reshaped(&dict, py, "bench_hp", batch.bench_hp, &bench)?;
    set_reshaped(&dict, py, "bench_present", batch.bench_present, &bench)?;
    set_reshaped(&dict, py, "switch_legal", batch.switch_legal, &bench)?;
    set_reshaped(&dict, py, "opp_bench_species", batch.opp_bench_species, &bench)?;
    set_reshaped(&dict, py, "opp_bench_hp", batch.opp_bench_hp, &bench)?;
    set_reshaped(&dict, py, "opp_bench_present", batch.opp_bench_present, &bench)?;

    // 3 次元 (B, NUM_BENCH, MAX_MOVE_SLOTS)
    let bench_moves = [b, NUM_BENCH, MAX_MOVE_SLOTS];
    set_reshaped(&dict, py, "bench_move_gids", batch.bench_move_gids, &bench_moves)?;
    set_reshaped(
        &dict,
        py,
        "bench_move_present",
        batch.bench_move_present,
        &bench_moves,
    )?;
    set_reshaped(
        &dict,
        py,
        "opp_bench_move_gids",
        batch.opp_bench_move_gids,
        &bench_moves,
    )?;
    set_reshaped(
        &dict,
        py,
        "opp_bench_move_present",
        batch.opp_bench_move_present,
        &bench_moves,
    )?;

    // (B, ACTION_DIM) bool
    set_reshaped(
        &dict,
        py,
        "legal_action_mask",
        batch.legal_action_mask,
        &[b, ACTION_DIM],
    )?;

    Ok(dict)
}

/// 学習ホットパス (`recv_observations`) 用: RootTask 側でパック済みのバッチを
/// numpy dict に包むだけ (パック実装と列レイアウトは `poke_ai3::packed` が唯一の正)。
pub fn packed_batch_to_dict<'py>(
    py: Python<'py>,
    batch: PackedBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let b = batch.rows;
    let dict = PyDict::new(py);
    set_reshaped(&dict, py, "packed_i64", batch.packed_i64, &[b, batch.ki])?;
    set_reshaped(&dict, py, "packed_f32", batch.packed_f32, &[b, batch.kf])?;
    set_reshaped(
        &dict,
        py,
        "legal_action_mask",
        batch.legal_action_mask,
        &[b, ACTION_DIM],
    )?;

    dict.set_item("game_id", batch.game_id.into_pyarray(py))?;
    dict.set_item("player", batch.player.into_pyarray(py))?;
    dict.set_item("request_id", batch.request_id.into_pyarray(py))?;
    dict.set_item("empty_game_id", batch.empty_game_id.into_pyarray(py))?;
    dict.set_item("empty_player", batch.empty_player.into_pyarray(py))?;
    dict.set_item("empty_request_id", batch.empty_request_id.into_pyarray(py))?;
    Ok(dict)
}

fn set_reshaped<'py, T: numpy::Element>(
    dict: &Bound<'py, PyDict>,
    py: Python<'py>,
    key: &str,
    data: Vec<T>,
    shape: &[usize],
) -> PyResult<()> {
    let array = data
        .into_pyarray(py)
        .reshape(shape.to_vec())
        .map_err(|e| PyRuntimeError::new_err(format!("reshape {key} failed: {e}")))?;
    dict.set_item(key, array)
}
