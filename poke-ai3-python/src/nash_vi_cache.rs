//! 遷移分布キャッシュ版ソルバ。
//!
//! `(s,a1,a2)→次状態分布` を一度だけ列挙して平坦配列 (CSR) に格納し、backward VI /
//! policy_eval / best_response の各反復で使い回す。毎反復のフル列挙 (crit×roll×coin の直積、
//! both-attack で ~2000 分岐) を「キャッシュ参照 + 2×2 Nash」に置換して数十倍高速化する。
//! 継続は `γ·V(s') + (1-γ)·tiebreak(s')` (幾何打ち切りゲーム。γ=1 で素の undiscounted)。

use poke_sho_rust::battle::{Choice, Player};
use poke_sho_rust::scenario::{MoveId, Stage};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::nash_vi_game::{Dims, Outcome, fill, legal, templates, tiebreak};
use crate::nash_vi_solve::solve_zero_sum;
use crate::nash_vi_trans::{TransCfg, transition};

const SENTINEL: u16 = 0xFFFF;
const SCALE: f32 = 1000.0;
/// 3d の完全方策スロット: Crunch / Dark Pulse / 現個体の弱点技 / Switch。
pub const FULL_ACTIONS: usize = 4;

fn action_code(choice: Choice) -> usize {
    match choice {
        Choice::Move(MoveId::Crunch) => 0,
        Choice::Move(MoveId::DarkPulse) => 1,
        Choice::Move(_) => 2,
        Choice::Switch(_) => 3,
    }
}

/// 全 valid 状態の遷移分布を平坦格納したキャッシュ。
pub struct TransCache {
    pub h: u64,
    pub total: usize,
    pub valid: Vec<u64>,        // vi -> global k
    pub nrows: Vec<u8>,         // vi -> P1 手数
    pub ncols: Vec<u8>,         // vi -> P2 手数
    pub row_sw: Vec<bool>,      // 平坦: 各 vi の行が交代か (len = Σ nrows)
    pub row_off: Vec<usize>,    // vi -> row_sw への offset (len valid+1)
    pub col_sw: Vec<bool>,      // 平坦: 各 vi の列が交代か (len = Σ ncols)
    pub row_action: Vec<u8>,    // 平坦: FULL_ACTIONS の意味スロット
    pub col_action: Vec<u8>,
    pub col_off: Vec<usize>,    // vi -> col_sw への offset
    pub cell_base: Vec<usize>,  // vi -> 最初のセル番号 (セル(vi,i,j)=cell_base[vi]+i*ncols+j)
    pub cell_off: Vec<usize>,   // セル番号 -> entries offset (len ncells+1)
    pub ek: Vec<u32>,           // index 帰結の global k2
    pub ew: Vec<f32>,           // index 帰結の確率重み
    pub term: Vec<f32>,         // セルごとの終端 (勝敗) 加重和
    pub tb: Vec<f32>,           // global k -> tiebreak 値 (size total)
}

struct CellBuild {
    idx: Vec<(u32, f32)>,
    term: f32,
}

struct StateBuild {
    row_sw: Vec<bool>,
    col_sw: Vec<bool>,
    row_action: Vec<u8>,
    col_action: Vec<u8>,
    cells: Vec<CellBuild>, // nrows*ncols, row-major
}

impl TransCache {
    pub fn build(stage: Stage, h: u64, cfg: &TransCfg, verbose: bool) -> Self {
        let total = (8 * h * h * h * h) as usize;
        let tmpl = templates(stage);
        let valid: Vec<u64> = (0..total as u64)
            .filter(|&k| Dims::decompose(k, h).active_alive())
            .collect();
        if verbose {
            println!("[cache] building H={h} valid={} ...", valid.len());
        }
        let builds: Vec<StateBuild> = valid
            .par_iter()
            .map(|&k| {
                let d = Dims::decompose(k, h);
                let st = fill(&tmpl[d.ai_team as usize], &d, h);
                let a1 = legal(&st, Player::P1);
                let a2 = legal(&st, Player::P2);
                let row_sw = a1.iter().map(|c| matches!(c, Choice::Switch(_))).collect();
                let col_sw = a2.iter().map(|c| matches!(c, Choice::Switch(_))).collect();
                let row_action = a1.iter().map(|&c| action_code(c) as u8).collect();
                let col_action = a2.iter().map(|&c| action_code(c) as u8).collect();
                let mut cells = Vec::with_capacity(a1.len() * a2.len());
                for &c1 in &a1 {
                    for &c2 in &a2 {
                        let tr = transition(&st, d.ai_team, c1, c2, cfg);
                        let mut idx = Vec::new();
                        let mut term = 0.0f32;
                        for (o, w) in tr {
                            match o {
                                Outcome::Index(k2) => idx.push((k2 as u32, w)),
                                Outcome::Terminal(t) => term += w * t,
                            }
                        }
                        cells.push(CellBuild { idx, term });
                    }
                }
                StateBuild { row_sw, col_sw, row_action, col_action, cells }
            })
            .collect();

        // 平坦化 (直列: offset を積算)。
        let n = valid.len();
        let mut nrows = Vec::with_capacity(n);
        let mut ncols = Vec::with_capacity(n);
        let mut row_sw = Vec::new();
        let mut row_off = Vec::with_capacity(n + 1);
        let mut col_sw = Vec::new();
        let mut row_action = Vec::new();
        let mut col_action = Vec::new();
        let mut col_off = Vec::with_capacity(n + 1);
        let mut cell_base = Vec::with_capacity(n + 1);
        let mut cell_off = vec![0usize];
        let mut ek = Vec::new();
        let mut ew = Vec::new();
        let mut term = Vec::new();
        for b in &builds {
            row_off.push(row_sw.len());
            col_off.push(col_sw.len());
            cell_base.push(term.len());
            nrows.push(b.row_sw.len() as u8);
            ncols.push(b.col_sw.len() as u8);
            row_sw.extend_from_slice(&b.row_sw);
            col_sw.extend_from_slice(&b.col_sw);
            row_action.extend_from_slice(&b.row_action);
            col_action.extend_from_slice(&b.col_action);
            for c in &b.cells {
                for &(k2, w) in &c.idx {
                    ek.push(k2);
                    ew.push(w);
                }
                cell_off.push(ek.len());
                term.push(c.term);
            }
        }
        row_off.push(row_sw.len());
        col_off.push(col_sw.len());
        cell_base.push(term.len());

        let mut tb = vec![0.0f32; total];
        for &k in &valid {
            tb[k as usize] = tiebreak(&Dims::decompose(k, h), h);
        }
        if verbose {
            let bytes = ek.len() * 8 + ew.len() * 4;
            println!("[cache] built: entries={} (~{:.1} GB)", ek.len(), bytes as f64 / 1e9);
        }
        TransCache {
            h, total, valid, nrows, ncols, row_sw, row_off, col_sw, col_off,
            row_action, col_action,
            cell_base, cell_off, ek, ew, term, tb,
        }
    }

    /// セル (vi,i,j) の期待値 M[i][j] を継続 v・割引 γ で評価。
    #[inline]
    fn cell_value(&self, cell: usize, v: &[f32], discount: f32) -> f32 {
        let (s, e) = (self.cell_off[cell], self.cell_off[cell + 1]);
        let mut acc = self.term[cell];
        for idx in s..e {
            let k2 = self.ek[idx] as usize;
            acc += self.ew[idx] * (discount * v[k2] + (1.0 - discount) * self.tb[k2]);
        }
        acc
    }

    /// 状態 vi のペイオフ行列を構築 (row-major)。
    fn matrix(&self, vi: usize, v: &[f32], discount: f32) -> Vec<Vec<f32>> {
        let (nr, nc) = (self.nrows[vi] as usize, self.ncols[vi] as usize);
        let base = self.cell_base[vi];
        (0..nr)
            .map(|i| (0..nc).map(|j| self.cell_value(base + i * nc + j, v, discount)).collect())
            .collect()
    }
}

fn strat_full(codes: &[u8], policy: &[f32], k: usize) -> Vec<f32> {
    codes.iter().map(|&code| policy[k * FULL_ACTIONS + code as usize]).collect()
}

fn solve_state_full(
    c: &TransCache,
    vi: usize,
    v: &[f32],
    discount: f32,
) -> (f32, [f32; FULL_ACTIONS]) {
    let m = c.matrix(vi, v, discount);
    let (val, strat) = solve_zero_sum(&m);
    let mut full = [0.0; FULL_ACTIONS];
    let off = c.row_off[vi];
    for (i, &p) in strat.iter().enumerate() {
        full[c.row_action[off + i] as usize] += p;
    }
    (val, full)
}

fn run_vi_full(
    c: &TransCache,
    discount: f32,
    horizon: u32,
    tol: f32,
    verbose: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut v = c.tb.clone();
    for layer in 1..=horizon {
        let nv: Vec<f32> = c.valid.par_iter().enumerate()
            .map(|(vi, _)| solve_state_full(c, vi, &v, discount).0).collect();
        let mut delta = 0.0f32;
        for (vi, &k) in c.valid.iter().enumerate() {
            delta = delta.max((nv[vi] - v[k as usize]).abs());
            v[k as usize] = nv[vi];
        }
        if verbose && (layer % 20 == 0 || delta < tol) {
            println!("[cache-full-vi] layer {layer}/{horizon}: max|ΔV|={delta:.2e}");
        }
        if delta < tol { break; }
    }
    let mut policy = vec![0.0f32; c.total * FULL_ACTIONS];
    let solved: Vec<[f32; FULL_ACTIONS]> = c.valid.par_iter().enumerate()
        .map(|(vi, _)| solve_state_full(c, vi, &v, discount).1).collect();
    for (vi, &k) in c.valid.iter().enumerate() {
        policy[k as usize * FULL_ACTIONS..(k as usize + 1) * FULL_ACTIONS]
            .copy_from_slice(&solved[vi]);
    }
    (v, policy)
}

fn eval_fixed_full(
    c: &TransCache,
    policy: &[f32],
    discount: f32,
    tol: f32,
    iters: u32,
    br: bool,
) -> Vec<f32> {
    let mut v = c.tb.clone();
    for _ in 0..iters {
        let nv: Vec<f32> = c.valid.par_iter().enumerate().map(|(vi, &k)| {
            let m = c.matrix(vi, &v, discount);
            let d = Dims::decompose(k, c.h);
            let mirror = d.swap().compose(c.h) as usize;
            let co = c.col_off[vi];
            let s2 = strat_full(&c.col_action[co..c.col_off[vi + 1]], policy, mirror);
            let rowval = |row: &[f32]| row.iter().zip(&s2).map(|(x, p)| x * p).sum::<f32>();
            if br {
                m.iter().map(|row| rowval(row)).fold(f32::NEG_INFINITY, f32::max)
            } else {
                let ro = c.row_off[vi];
                let s1 = strat_full(&c.row_action[ro..c.row_off[vi + 1]], policy, k as usize);
                m.iter().zip(&s1).map(|(row, p)| rowval(row) * p).sum()
            }
        }).collect();
        let mut delta = 0.0f32;
        for (vi, &k) in c.valid.iter().enumerate() {
            delta = delta.max((nv[vi] - v[k as usize]).abs());
            v[k as usize] = nv[vi];
        }
        if delta < tol { break; }
    }
    v
}

/// 交代マスクと P(交代) から着手分布 (技均等・交代質量を交代手に等分)。
fn strat_mask(mask: &[bool], pswitch: f32) -> Vec<f32> {
    let n_sw = mask.iter().filter(|&&b| b).count();
    let n = mask.len();
    if n_sw == 0 || n_sw == n {
        return vec![1.0 / n as f32; n];
    }
    let n_mv = n - n_sw;
    mask.iter()
        .map(|&b| if b { pswitch / n_sw as f32 } else { (1.0 - pswitch) / n_mv as f32 })
        .collect()
}

/// 状態 vi を解く: (V, P(交代))。
fn solve_state(c: &TransCache, vi: usize, v: &[f32], discount: f32) -> (f32, f32) {
    let m = c.matrix(vi, v, discount);
    let (val, strat) = solve_zero_sum(&m);
    let ro = c.row_off[vi];
    let pswitch: f32 = strat.iter().enumerate()
        .filter(|(i, _)| c.row_sw[ro + i])
        .map(|(_, s)| *s)
        .sum();
    (val, pswitch)
}

/// backward VI を horizon 層 (または収束) 掃く。返り (V, pswitch_full)。
fn run_vi(c: &TransCache, discount: f32, horizon: u32, tol: f32, verbose: bool) -> (Vec<f32>, Vec<f32>) {
    let mut v = c.tb.clone();
    for layer in 1..=horizon {
        let nv: Vec<f32> = c.valid.par_iter().enumerate()
            .map(|(vi, _)| solve_state(c, vi, &v, discount).0).collect();
        let mut delta = 0.0f32;
        for (vi, &k) in c.valid.iter().enumerate() {
            delta = delta.max((nv[vi] - v[k as usize]).abs());
            v[k as usize] = nv[vi];
        }
        if verbose && (layer % 20 == 0 || delta < tol) {
            println!("[cache-vi] layer {layer}/{horizon}: max|ΔV|={delta:.2e}");
        }
        if delta < tol {
            if verbose { println!("[cache-vi] converged at layer {layer}"); }
            break;
        }
    }
    let mut ps = vec![0.0f32; c.total];
    let sol: Vec<f32> = c.valid.par_iter().enumerate()
        .map(|(vi, _)| solve_state(c, vi, &v, discount).1).collect();
    for (vi, &k) in c.valid.iter().enumerate() {
        ps[k as usize] = sol[vi];
    }
    (v, ps)
}

/// 固定 σ の値 (方策評価) または P1 best-response 値を反復で求める。br=true で best-response。
fn eval_fixed(c: &TransCache, ps: &[f32], discount: f32, tol: f32, iters: u32, br: bool) -> Vec<f32> {
    let mut v = c.tb.clone();
    for _ in 0..iters {
        let nv: Vec<f32> = c.valid.par_iter().enumerate()
            .map(|(vi, &k)| {
                let m = c.matrix(vi, &v, discount);
                let d = Dims::decompose(k, c.h);
                let co = c.col_off[vi];
                let s2 = strat_mask(&c.col_sw[co..c.col_off[vi + 1]], ps[d.swap().compose(c.h) as usize]);
                let rowval = |row: &[f32]| row.iter().zip(&s2).map(|(mij, sj)| mij * sj).sum::<f32>();
                if br {
                    m.iter().map(|r| rowval(r)).fold(f32::NEG_INFINITY, f32::max)
                } else {
                    let ro = c.row_off[vi];
                    let s1 = strat_mask(&c.row_sw[ro..c.row_off[vi + 1]], ps[k as usize]);
                    m.iter().zip(&s1).map(|(r, si)| si * rowval(r)).sum::<f32>()
                }
            }).collect();
        let mut delta = 0.0f32;
        for (vi, &k) in c.valid.iter().enumerate() {
            delta = delta.max((nv[vi] - v[k as usize]).abs());
            v[k as usize] = nv[vi];
        }
        if delta < tol { break; }
    }
    v
}

/// 配信実物ゲーム (打ち切り無し=100手で引き分け) で、固定テーブル σ に対する P1 best-response を
/// 有限ホライズン (draw_value 種) で解く。P2 は table_ps を指す。返り (BR価値, BRの交代確率)*1000。
/// これで「時計を見られないテーブル」が実ゲームで本当に付け入られるかを正確に測る。
#[pyfunction]
#[pyo3(signature = (stage, hp_buckets, table_ps, crit=true, randomize=true,
                    draw_value=0.5, horizon=100, seed_tiebreak=false, verbose=true))]
#[allow(clippy::too_many_arguments)]
pub fn best_response_vs_table(
    stage: &str,
    hp_buckets: u64,
    table_ps: Vec<u16>,
    crit: bool,
    randomize: bool,
    draw_value: f32,
    horizon: u32,
    seed_tiebreak: bool,
    verbose: bool,
) -> PyResult<(Vec<u16>, Vec<u16>)> {
    let stage = Stage::from_short_name(stage)
        .ok_or_else(|| PyValueError::new_err("unknown stage"))?;
    let cfg = TransCfg { h: hp_buckets, crit, randomize };
    let c = TransCache::build(stage, hp_buckets, &cfg, verbose);
    let ps: Vec<f32> = table_ps.iter().map(|&x| x as f32 / SCALE).collect();
    if ps.len() != c.total {
        return Err(PyValueError::new_err("table_ps length != total"));
    }
    // 100手到達時の種: seed_tiebreak なら残HPタイブレーク、そうでなければ draw_value(引き分け)。
    // P1 は最大化、P2 は固定 σ、割引なし。
    let mut v = vec![0.0f32; c.total];
    for &k in &c.valid {
        v[k as usize] = if seed_tiebreak { c.tb[k as usize] } else { draw_value };
    }
    let mut br_row_switch = vec![0.0f32; c.total];
    for layer in 1..=horizon {
        let res: Vec<(f32, f32)> = c.valid.par_iter().enumerate()
            .map(|(vi, &k)| {
                let (nr, nc) = (c.nrows[vi] as usize, c.ncols[vi] as usize);
                let base = c.cell_base[vi];
                let d = Dims::decompose(k, c.h);
                let co = c.col_off[vi];
                let s2 = strat_mask(&c.col_sw[co..c.col_off[vi + 1]], ps[d.swap().compose(c.h) as usize]);
                let ro = c.row_off[vi];
                // P1 の各行 i の期待値 Σ_j s2[j]·M[i][j]、最大の行を採用 (割引=1 で cont=v[k2])。
                let mut best_val = f32::NEG_INFINITY;
                let mut best_sw = 0.0f32;
                for i in 0..nr {
                    let mut rv = 0.0f32;
                    for j in 0..nc {
                        rv += s2[j] * c.cell_value(base + i * nc + j, &v, 1.0);
                    }
                    if rv > best_val {
                        best_val = rv;
                        best_sw = if c.row_sw[ro + i] { 1.0 } else { 0.0 };
                    }
                }
                (best_val, best_sw)
            }).collect();
        let mut delta = 0.0f32;
        for (vi, &k) in c.valid.iter().enumerate() {
            delta = delta.max((res[vi].0 - v[k as usize]).abs());
            v[k as usize] = res[vi].0;
            br_row_switch[k as usize] = res[vi].1;
        }
        if verbose && (layer % 10 == 0 || layer == horizon) {
            println!("[br-vs-table] layer {layer}/{horizon}: max|ΔV|={delta:.2e}");
        }
    }
    let mut brv = vec![SENTINEL; c.total];
    let mut brp = vec![SENTINEL; c.total];
    for &k in &c.valid {
        brv[k as usize] = (v[k as usize] * SCALE).round().clamp(0.0, SCALE) as u16;
        brp[k as usize] = (br_row_switch[k as usize] * SCALE).round() as u16;
    }
    Ok((brv, brp))
}

/// キャッシュ版 b&c/幾何打ち切り Nash。web 形式 (P(交代), V, BR)*1000 u16 を返す。
#[pyfunction]
#[pyo3(signature = (stage, hp_buckets, crit=true, randomize=true, horizon=3000,
                    discount=0.99, tol=1e-6, eval_iters=3000, verbose=true))]
#[allow(clippy::too_many_arguments)]
pub fn solve_nash_cached(
    stage: &str,
    hp_buckets: u64,
    crit: bool,
    randomize: bool,
    horizon: u32,
    discount: f32,
    tol: f32,
    eval_iters: u32,
    verbose: bool,
) -> PyResult<(Vec<u16>, Vec<u16>, Vec<u16>, f32, f32)> {
    let stage = Stage::from_short_name(stage)
        .ok_or_else(|| PyValueError::new_err("unknown stage"))?;
    if !stage.is_party() {
        return Err(PyValueError::new_err("needs a party stage (3b)"));
    }
    let cfg = TransCfg { h: hp_buckets, crit, randomize };
    let c = TransCache::build(stage, hp_buckets, &cfg, verbose);
    let (_v, ps) = run_vi(&c, discount, horizon, tol, verbose);
    let v_true = eval_fixed(&c, &ps, discount, tol, eval_iters, false);
    let v_br = eval_fixed(&c, &ps, discount, tol, eval_iters, true);
    let (mut em, mut ex) = (0.0f32, 0.0f32);
    for &k in &c.valid {
        let g = (v_br[k as usize] - v_true[k as usize]).max(0.0);
        em += g;
        ex = ex.max(g);
    }
    em /= c.valid.len() as f32;
    if verbose {
        println!("[cache] exploitability mean={em:.5} max={ex:.5}");
    }
    let mut policy = vec![SENTINEL; c.total];
    let mut value = vec![SENTINEL; c.total];
    let mut br = vec![SENTINEL; c.total];
    for &k in &c.valid {
        let ki = k as usize;
        policy[ki] = (ps[ki] * SCALE).round().clamp(0.0, SCALE) as u16;
        value[ki] = (v_true[ki] * SCALE).round().clamp(0.0, SCALE) as u16;
        br[ki] = (v_br[ki] * SCALE).round().clamp(0.0, SCALE) as u16;
    }
    Ok((policy, value, br, em, ex))
}

/// 多技 party stage 用。方策は状態ごとに
/// [Crunch, Dark Pulse, 弱点技, Switch] の4要素を持つ平坦配列。
#[pyfunction]
#[pyo3(signature = (stage, hp_buckets, crit=true, randomize=true, horizon=3000,
                    discount=0.99, vi_tol=5e-6, eval_tol=1e-6, eval_iters=3000,
                    verbose=true))]
#[allow(clippy::too_many_arguments)]
pub fn solve_nash_cached_full(
    stage: &str,
    hp_buckets: u64,
    crit: bool,
    randomize: bool,
    horizon: u32,
    discount: f32,
    vi_tol: f32,
    eval_tol: f32,
    eval_iters: u32,
    verbose: bool,
) -> PyResult<(Vec<u16>, Vec<u16>, Vec<u16>, f32, f32)> {
    let stage = Stage::from_short_name(stage)
        .ok_or_else(|| PyValueError::new_err("unknown stage"))?;
    if !stage.is_party() {
        return Err(PyValueError::new_err("needs a party stage"));
    }
    let cfg = TransCfg { h: hp_buckets, crit, randomize };
    let c = TransCache::build(stage, hp_buckets, &cfg, verbose);
    let (_v, policy) = run_vi_full(&c, discount, horizon, vi_tol, verbose);
    let v_true = eval_fixed_full(&c, &policy, discount, eval_tol, eval_iters, false);
    let v_br = eval_fixed_full(&c, &policy, discount, eval_tol, eval_iters, true);
    let (mut em, mut ex) = (0.0f32, 0.0f32);
    for &k in &c.valid {
        let gap = (v_br[k as usize] - v_true[k as usize]).max(0.0);
        em += gap;
        ex = ex.max(gap);
    }
    em /= c.valid.len() as f32;
    if verbose {
        println!("[cache-full] exploitability mean={em:.5} max={ex:.5}");
    }
    let mut out_policy = vec![SENTINEL; c.total * FULL_ACTIONS];
    let mut value = vec![SENTINEL; c.total];
    let mut br_value = vec![SENTINEL; c.total];
    for &k in &c.valid {
        let ki = k as usize;
        for a in 0..FULL_ACTIONS {
            out_policy[ki * FULL_ACTIONS + a] =
                (policy[ki * FULL_ACTIONS + a] * SCALE).round().clamp(0.0, SCALE) as u16;
        }
        value[ki] = (v_true[ki] * SCALE).round().clamp(0.0, SCALE) as u16;
        br_value[ki] = (v_br[ki] * SCALE).round().clamp(0.0, SCALE) as u16;
    }
    Ok((out_policy, value, br_value, em, ex))
}
