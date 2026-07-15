//! 有限ホライズン backward induction (b&c)。
//!
//! 実ゲームは 100 手で打ち切り。turn は状態に含めないが、終端を「引分 0.5」でなく残存
//! HP タイブレーク (`nash_vi_game::tiebreak`) にすることで、turn 非依存の状態だけで終端値が
//! 確定する。継続を割引なし (γ=1) で後ろ向きに掃くと、有利側が最終手に決着させる戦略が
//! backward に unravel し、割引 VI の stall 縮退が解ける。得られる σ は実ゲームの (定常近似)
//! Nash で、on-path は 100 手手前で決着するため大ホライズン層を web に配信すればよい。

use poke_sho_rust::scenario::Stage;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::nash_vi_eval::{best_response, policy_eval};
use crate::nash_vi_game::{Dims, tiebreak, templates};
use crate::nash_vi_solve::solve_state;
use crate::nash_vi_trans::TransCfg;

const SENTINEL: u16 = 0xFFFF;
const SCALE: f32 = 1000.0;

/// 終端 (0 手残り = 100 手到達) の値ベクトル: valid はタイブレーク、無効は 0。
fn tiebreak_vec(h: u64, total: usize, valid: &[u64]) -> Vec<f32> {
    let mut v = vec![0.0f32; total];
    for &k in valid {
        v[k as usize] = tiebreak(&Dims::decompose(k, h), h);
    }
    v
}

/// 割引なし (γ=1) の backward induction を horizon 層 (または収束まで) 掃く。
/// V は「残り手数 → 各状態の値」を層ごとに更新し、最終層 (≒大ホライズン) を返す。
fn run_backward(
    h: u64,
    cfg: &TransCfg,
    tmpl: &[poke_sho_rust::battle::BattleState; 2],
    horizon: u32,
    discount: f32,
    tol: f32,
    verbose: bool,
) -> (Vec<f32>, Vec<u64>) {
    let total = (8 * h * h * h * h) as usize;
    let valid: Vec<u64> = (0..total as u64)
        .filter(|&k| Dims::decompose(k, h).active_alive())
        .collect();
    if verbose {
        println!("[nash-back] H={h} total={total} valid={} horizon={horizon}", valid.len());
    }
    let mut v = tiebreak_vec(h, total, &valid);
    if verbose {
        let mut e = 0.0f32;
        for &k in &valid {
            let sk = Dims::decompose(k, h).swap().compose(h) as usize;
            e = e.max((v[k as usize] + v[sk] - 1.0).abs());
        }
        println!("[nash-back] tiebreak seed max|V(s)+V(swap)-1|={e:.2e}");
    }
    for layer in 1..=horizon {
        let newvals: Vec<f32> = valid
            .par_iter()
            .map(|&k| solve_state(k, &v, tmpl, cfg, discount).0)
            .collect();
        let mut delta = 0.0f32;
        for (i, &k) in valid.iter().enumerate() {
            delta = delta.max((newvals[i] - v[k as usize]).abs());
            v[k as usize] = newvals[i];
        }
        if verbose && (layer % 10 == 0 || delta < tol) {
            println!("[nash-back] layer {layer}/{horizon}: max|ΔV|={delta:.2e}");
        }
        if delta < tol {
            if verbose {
                println!("[nash-back] converged at layer {layer} (stationary limit)");
            }
            break;
        }
    }
    (v, valid)
}

/// 計測用: 各層 (残り手数 t=1..=horizon) の σ (P(交代)*1000, u16) を全部返す。
/// 返り値 [layer][index]。exact な turn 依存均衡がどれだけ層間で変わるかの定量化に使う。
#[pyfunction]
#[pyo3(signature = (stage, hp_buckets, crit=true, randomize=true, horizon=100))]
pub fn solve_nash_layers(
    stage: &str,
    hp_buckets: u64,
    crit: bool,
    randomize: bool,
    horizon: u32,
) -> PyResult<Vec<Vec<u16>>> {
    let stage = Stage::from_short_name(stage)
        .ok_or_else(|| PyValueError::new_err("unknown stage"))?;
    let h = hp_buckets;
    let total = (8 * h * h * h * h) as usize;
    let cfg = TransCfg { h, crit, randomize };
    let tmpl = templates(stage);
    let valid: Vec<u64> = (0..total as u64)
        .filter(|&k| Dims::decompose(k, h).active_alive())
        .collect();
    let mut v = tiebreak_vec(h, total, &valid);
    let mut layers: Vec<Vec<u16>> = Vec::with_capacity(horizon as usize);
    for _ in 1..=horizon {
        let solved: Vec<(f32, f32)> = valid
            .par_iter()
            .map(|&k| solve_state(k, &v, &tmpl, &cfg, 1.0))
            .collect();
        let mut pol = vec![SENTINEL; total];
        for (i, &k) in valid.iter().enumerate() {
            v[k as usize] = solved[i].0;
            pol[k as usize] = (solved[i].1 * SCALE).round().clamp(0.0, SCALE) as u16;
        }
        layers.push(pol);
    }
    Ok(layers)
}

/// b&c backward induction を解いて web 形式の (P(交代)*1000, V*1000, BR*1000) を u16 で返す。
/// 割引 VI (`solve_nash_vi`) の turn 非依存×引分終端の不整合を、HP タイブレーク終端で解消した版。
#[pyfunction]
#[pyo3(signature = (stage, hp_buckets, crit=true, randomize=true, horizon=100,
                    discount=1.0, tol=1e-6, eval_iters=4000, verbose=true))]
#[allow(clippy::too_many_arguments)]
pub fn solve_nash_backward(
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
        return Err(PyValueError::new_err("nash backward needs a party stage (3b)"));
    }
    let h = hp_buckets;
    let total = (8 * h * h * h * h) as usize;
    let cfg = TransCfg { h, crit, randomize };
    let tmpl = templates(stage);

    // フェーズ1: backward induction で戦略 σ (=P(交代)) と定常値 V を得る。
    let (v_disc, valid) = run_backward(h, &cfg, &tmpl, horizon, discount, tol, verbose);
    // σ 抽出も同じ割引作用素で (微小割引がタイを決着方向に破り、弱支配される行を選ばない)。
    let mut pswitch_full = vec![0.0f32; total];
    let strat: Vec<(u64, f32)> = valid
        .par_iter()
        .map(|&k| (k, solve_state(k, &v_disc, &tmpl, &cfg, discount).1))
        .collect();
    for (k, ps) in strat {
        pswitch_full[k as usize] = ps;
    }

    if verbose {
        // 生 backward V の零和鏡像整合 V(s)+V(swap(s))=1 を検査 (破れていれば σ が非対称)。
        let mut sym_err = 0.0f32;
        let mut worst = 0u64;
        for &k in &valid {
            let sk = Dims::decompose(k, h).swap().compose(h) as usize;
            let e = (v_disc[k as usize] + v_disc[sk] - 1.0).abs();
            if e > sym_err { sym_err = e; worst = k; }
        }
        let wd = Dims::decompose(worst, h);
        let ws = wd.swap().compose(h);
        println!("[nash-back] worst sym state k={worst} {wd:?} V={:.4} swapV={:.4}",
            v_disc[worst as usize], v_disc[ws as usize]);
        let (m, _) = crate::nash_vi_solve::build_matrix(worst, &v_disc, &tmpl, &cfg, 1.0);
        let ( m2, _) = crate::nash_vi_solve::build_matrix(ws, &v_disc, &tmpl, &cfg, 1.0);
        println!("[nash-back]   M(s)={m:?}");
        println!("[nash-back]   M(swap)={m2:?}");
        let f = |a: u64, e: u64| (((((a * 2) * h + (h - 1)) * h + (h - 1)) * 2 + e) * h + (h - 1)) * h + (h - 1);
        let (t0, t1) = (f(0, 0) as usize, f(1, 0) as usize);
        println!("[nash-back] raw V start team0={:.4} team1={:.4} sum={:.4} | max|V(s)+V(swap)-1|={:.2e}",
            v_disc[t0], v_disc[t1], v_disc[t0] + v_disc[t1], sym_err);
    }
    // exploitability は σ を解いたのと同じゲーム (discount<1 なら幾何打ち切り) の作用素で測る。
    // 幾何打ち切りゲームは memoryless なので定常 σ が厳密 Nash になれる (=BR gap ≈ 0 を検証)。
    // 真値 V = σ 同士の方策評価、BR = P1 best-response (どちらも縮小写像で幾何収束)。
    let tb = tiebreak_vec(h, total, &valid);
    let v_true = policy_eval(&tb, &valid, &pswitch_full, &tmpl, &cfg, discount, tol, eval_iters);
    let v_br = best_response(&v_true, &valid, &pswitch_full, &tmpl, &cfg, discount, tol, eval_iters);
    let mut expl_max = 0.0f32;
    let mut expl_sum = 0.0f32;
    let mut worst = 0u64;
    for &k in &valid {
        let g = (v_br[k as usize] - v_true[k as usize]).max(0.0);
        if g > expl_max { expl_max = g; worst = k; }
        expl_sum += g;
    }
    let expl_mean = expl_sum / valid.len() as f32;
    if verbose {
        println!("[nash-back] finite-horizon exploitability: mean={expl_mean:.4} max={expl_max:.4}");
        let wd = Dims::decompose(worst, h);
        println!("[nash-back] worst exploit k={worst} {wd:?} V={:.3} BR={:.3} pswitch={:.3}",
            v_true[worst as usize], v_br[worst as usize], pswitch_full[worst as usize]);
        let (m, a1) = crate::nash_vi_solve::build_matrix(worst, &v_disc, &tmpl, &cfg, 1.0);
        println!("[nash-back]   a1={a1:?} M={m:?}");
        let sk = wd.swap().compose(h);
        println!("[nash-back]   P2 σ from swap k={sk} pswitch={:.3}", pswitch_full[sk as usize]);
    }

    let mut policy = vec![SENTINEL; total];
    let mut value = vec![SENTINEL; total];
    let mut br = vec![SENTINEL; total];
    for &k in &valid {
        let ki = k as usize;
        policy[ki] = (pswitch_full[ki] * SCALE).round().clamp(0.0, SCALE) as u16;
        value[ki] = (v_true[ki] * SCALE).round().clamp(0.0, SCALE) as u16;
        br[ki] = (v_br[ki] * SCALE).round().clamp(0.0, SCALE) as u16;
    }
    Ok((policy, value, br, expl_mean, expl_max))
}
