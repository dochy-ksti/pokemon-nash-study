mod encoded;
mod human_game;
mod nash_vi_back;
mod nash_vi_cache;
mod nash_vi_eval;
mod nash_vi_game;
mod nash_vi_solve;
mod nash_vi_trans;
mod policy_table;

use encoded::{encode_observation_states, packed_batch_to_dict};
use nash_vi_back::{solve_nash_backward, solve_nash_layers};
use nash_vi_cache::{best_response_vs_table, solve_nash_cached};
use nash_vi_solve::{debug_nash_matrices, solve_nash_vi};
use policy_table::enumerate_policy_batch;
use poke_ai3::packed::packed_layout_json;
use human_game::PyHumanGame;
use numpy::PyReadonlyArray1;
use poke_ai3::{Backend, LookaheadConfig, RustAsyncExecutor};
use poke_env_rust::observation::{ACTION_DIM, MAX_MOVE_SLOTS, NUM_BENCH, NUM_MOVES, Stage};
use poke_sho_rust::global_ids;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "RustAsyncExecutor")]
pub struct PyRustAsyncExecutor {
    inner: RustAsyncExecutor,
}

#[pymethods]
impl PyRustAsyncExecutor {
    #[new]
    #[pyo3(signature = (num_games, max_batch_size=None, trajectories_threshold=None, backend="local", randomize=true, crit_enabled=true, stage="3b", sims=64, sim_concurrency=1, search_turn_min=4, search_turn_max=8, eval_rule_opponent=false, eval_rule_p1=false, battle_seed=1, nash_learning_rate=1.5, nash_weak=true, depth_skew=1.0, policy_only=false, value_target_expected=false, max_turns=100))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        num_games: usize,
        max_batch_size: Option<usize>,
        trajectories_threshold: Option<usize>,
        backend: &str,
        randomize: bool,
        crit_enabled: bool,
        stage: &str,
        sims: u32,
        sim_concurrency: u32,
        search_turn_min: u32,
        search_turn_max: u32,
        eval_rule_opponent: bool,
        eval_rule_p1: bool,
        battle_seed: u64,
        nash_learning_rate: f32,
        nash_weak: bool,
        depth_skew: f32,
        policy_only: bool,
        value_target_expected: bool,
        max_turns: u32,
    ) -> PyResult<Self> {
        // 実験 (experiments/poke-ai3) で max_batch_size≈num_games*W/2 がスループット最適と判明。
        // num_games*W だとバリアが深くパイプラインが途切れて最遅になる。
        let max_batch_size = max_batch_size
            .unwrap_or((num_games * sim_concurrency.max(1) as usize / 2).max(1));
        let trajectories_threshold = trajectories_threshold.unwrap_or(num_games);
        if num_games == 0 || max_batch_size == 0 || trajectories_threshold == 0 {
            return Err(PyValueError::new_err(
                "num_games, max_batch_size, and trajectories_threshold must be positive",
            ));
        }
        if sims == 0 || search_turn_min == 0 || search_turn_max < search_turn_min {
            return Err(PyValueError::new_err(
                "sims must be > 0 and 0 < search_turn_min <= search_turn_max",
            ));
        }
        if sim_concurrency == 0 || sim_concurrency > sims {
            return Err(PyValueError::new_err(
                "sim_concurrency must be > 0 and <= sims",
            ));
        }
        let backend = parse_backend(backend)?;
        let stage = parse_stage(stage)?;
        let lookahead = LookaheadConfig {
            sims,
            sim_concurrency,
            search_turn_min,
            search_turn_max,
            depth_skew,
            policy_only,
            randomize,
            crit_enabled,
            nash_learning_rate,
            nash_weak,
            value_target_expected,
            ..LookaheadConfig::default()
        };
        Ok(Self {
            inner: RustAsyncExecutor::new(
                num_games,
                battle_seed,
                max_batch_size,
                trajectories_threshold,
                backend,
                randomize,
                crit_enabled,
                stage,
                max_turns,
                lookahead,
                eval_rule_opponent,
                eval_rule_p1,
            ),
        })
    }

    fn is_ready(&mut self) -> bool {
        self.inner.is_ready()
    }

    fn trajectories_ready(&mut self) -> bool {
        self.inner.trajectories_ready()
    }

    /// 観測バッチをパック形式 (packed_i64 / packed_f32 / legal_action_mask +
    /// ルーティング/empty 配列) の numpy dict で受け取る。列レイアウトは
    /// `ENCODED_PACKED_LAYOUT` (JSON) を唯一の正とする。
    fn recv_observations<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let batch = self
            .inner
            .recv_packed()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        packed_batch_to_dict(py, batch)
    }

    /// 推論結果を配列で返送する。policy は (B*ACTION_DIM,) または (B, ACTION_DIM) を
    /// ravel した連続配列、その他は (B,)。empty_* は recv_observations の同名配列を
    /// そのままエコーバックする。
    #[allow(clippy::too_many_arguments)]
    fn send_inference(
        &self,
        game_id: PyReadonlyArray1<'_, i64>,
        player: PyReadonlyArray1<'_, i64>,
        request_id: PyReadonlyArray1<'_, i64>,
        policy: PyReadonlyArray1<'_, f32>,
        value: PyReadonlyArray1<'_, f32>,
        empty_game_id: PyReadonlyArray1<'_, i64>,
        empty_player: PyReadonlyArray1<'_, i64>,
        empty_request_id: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        self.inner
            .send_inference(
                game_id.as_slice()?,
                player.as_slice()?,
                request_id.as_slice()?,
                policy.as_slice()?,
                value.as_slice()?,
                empty_game_id.as_slice()?,
                empty_player.as_slice()?,
                empty_request_id.as_slice()?,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn recv_trajectories(&mut self) -> PyResult<String> {
        self.inner
            .recv_trajectories_json()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// 敵混合学習用の役割テーブルを更新する。長さは num_games に一致させる。
    /// roles[game_id]=0 で自己対戦、非0 で敵ゲーム (P2 policy-only + enemy_game タグ)。
    fn set_roles(&self, roles: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        self.inner
            .set_roles(roles.as_slice()?)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

}

#[pyfunction]
#[pyo3(signature = (num_games, max_batch_size=None, trajectories_threshold=None, backend="local", randomize=true, crit_enabled=true, stage="3b", sims=64, sim_concurrency=1, search_turn_min=4, search_turn_max=8, eval_rule_opponent=false, eval_rule_p1=false, battle_seed=1, nash_learning_rate=1.5, nash_weak=true, depth_skew=1.0, policy_only=false, value_target_expected=false, max_turns=100))]
#[allow(clippy::too_many_arguments)]
pub fn get_rust_async_executor_wrapper(
    num_games: usize,
    max_batch_size: Option<usize>,
    trajectories_threshold: Option<usize>,
    backend: &str,
    randomize: bool,
    crit_enabled: bool,
    stage: &str,
    sims: u32,
    sim_concurrency: u32,
    search_turn_min: u32,
    search_turn_max: u32,
    eval_rule_opponent: bool,
    eval_rule_p1: bool,
    battle_seed: u64,
    nash_learning_rate: f32,
    nash_weak: bool,
    depth_skew: f32,
    policy_only: bool,
    value_target_expected: bool,
    max_turns: u32,
) -> PyResult<PyRustAsyncExecutor> {
    PyRustAsyncExecutor::py_new(
        num_games,
        max_batch_size,
        trajectories_threshold,
        backend,
        randomize,
        crit_enabled,
        stage,
        sims,
        sim_concurrency,
        search_turn_min,
        search_turn_max,
        eval_rule_opponent,
        eval_rule_p1,
        battle_seed,
        nash_learning_rate,
        nash_weak,
        depth_skew,
        policy_only,
        value_target_expected,
        max_turns,
    )
}

fn parse_backend(name: &str) -> PyResult<Backend> {
    match name {
        "local" => Ok(Backend::Local),
        "showdown" => Ok(Backend::Showdown),
        other => Err(PyValueError::new_err(format!(
            "unknown backend '{}'; expected 'local' or 'showdown'",
            other
        ))),
    }
}

fn parse_stage(name: &str) -> PyResult<Stage> {
    Stage::from_short_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown stage '{}'; expected '3a' or '3b'",
            name
        ))
    })
}

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRustAsyncExecutor>()?;
    module.add_class::<PyHumanGame>()?;
    module.add_function(wrap_pyfunction!(get_rust_async_executor_wrapper, module)?)?;
    module.add_function(wrap_pyfunction!(encode_observation_states, module)?)?;
    module.add_function(wrap_pyfunction!(enumerate_policy_batch, module)?)?;
    module.add_function(wrap_pyfunction!(solve_nash_vi, module)?)?;
    module.add_function(wrap_pyfunction!(solve_nash_backward, module)?)?;
    module.add_function(wrap_pyfunction!(solve_nash_layers, module)?)?;
    module.add_function(wrap_pyfunction!(solve_nash_cached, module)?)?;
    module.add_function(wrap_pyfunction!(best_response_vs_table, module)?)?;
    module.add_function(wrap_pyfunction!(debug_nash_matrices, module)?)?;
    // 観測・行動空間の定数とグローバル ID 表。Python 側 (encoding.py) はここから
    // import し、Rust の定義を唯一の正とする。
    module.add("NUM_MOVES", NUM_MOVES)?;
    module.add("MAX_MOVE_SLOTS", MAX_MOVE_SLOTS)?;
    module.add("NUM_BENCH", NUM_BENCH)?;
    module.add("ACTION_DIM", ACTION_DIM)?;
    let (species_vocab, move_vocab, type_vocab) = global_ids::vocab_sizes();
    module.add("SPECIES_VOCAB", species_vocab)?;
    module.add("MOVE_VOCAB", move_vocab)?;
    module.add("TYPE_VOCAB", type_vocab)?;
    // Embedding 固定容量 (語彙上限)。checkpoint に焼き付く Embedding の行数はこの容量で
    // 固定し、TSV 追記 (新技・新種族) で形状が変わらないようにする。
    module.add("SPECIES_VOCAB_CAP", global_ids::SPECIES_VOCAB_CAP)?;
    module.add("MOVE_VOCAB_CAP", global_ids::MOVE_VOCAB_CAP)?;
    module.add("ENCODED_PACKED_LAYOUT", packed_layout_json())?;
    module.add("SPECIES_TABLE_TSV", global_ids::species_tsv())?;
    module.add("MOVE_TABLE_TSV", global_ids::move_tsv())?;
    module.add("TYPE_TABLE_TSV", global_ids::type_tsv())?;
    Ok(())
}
