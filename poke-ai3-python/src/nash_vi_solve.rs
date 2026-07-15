//! 2人ゼロ和行列ゲーム解法 + Nash 値反復 (Shapley) ドライバ。
//!
//! 各状態 s で P1(AI)=行 (最大化)・P2=列 (最小化) の行列 `M[i][j] = E_遷移[継続値]` を
//! 作り、ゼロ和ナッシュ値 `V(s)` と P1 混合戦略 σ(s) を求める。全状態を収束まで掃く
//! (Jacobi・rayon 並列)。継続値は `γ·V(s') + (1-γ)·0.5` とし、`(1-γ)` を「ターン上限で
//! 引分(0.5)」の幾何近似とみなす (γ=0.99 で平均 100 ターン ≒ 実 MAX_TURNS)。これで
//! 純交代ループも 0.5 へ安定収束し、対称局面の値は 0.5 に固定される。

use poke_sho_rust::battle::Choice;
use poke_sho_rust::battle::Player;
use poke_sho_rust::scenario::Stage;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::nash_vi_eval::{best_response, policy_eval};
use crate::nash_vi_game::{Dims, Outcome, fill, legal, templates};
use crate::nash_vi_trans::{TransCfg, transition};

const SENTINEL: u16 = 0xFFFF;
const SCALE: f32 = 1000.0;

/// ゼロ和行列 (row=最大化) のゲーム値と行(P1)混合戦略を返す。
/// 1×n / n×1 と 2×2 は解析的に、それ以外は fictitious play で解く。
pub(crate) fn solve_zero_sum(m: &[Vec<f32>]) -> (f32, Vec<f32>) {
    let rows = m.len();
    let cols = if rows > 0 { m[0].len() } else { 0 };
    if rows == 0 || cols == 0 {
        return (0.5, vec![]);
    }
    if rows == 1 {
        let v = m[0].iter().cloned().fold(f32::INFINITY, f32::min);
        return (v, vec![1.0]);
    }
    if cols == 1 {
        let mut best = 0usize;
        for i in 1..rows {
            if m[i][0] > m[best][0] {
                best = i;
            }
        }
        let mut s = vec![0.0; rows];
        s[best] = 1.0;
        return (m[best][0], s);
    }
    // 純戦略の鞍点判定。零和では鞍点⇔ v_lower(maximin)==v_upper(minimax)。数値ノイズと
    // 退化(denom≈0)を安全に扱うため、gap がしきい値未満なら純戦略とみなす (誤差 < gap で
    // nonexpansive)。gap がしきい値以上のときだけ 2×2 混合式 (denom は 0 から十分離れる) や
    // FP を使う。これで VI 作用素が確実に γ-縮小になる。
    const SADDLE_EPS: f32 = 1e-7;
    let row_min: Vec<f32> = m.iter()
        .map(|r| r.iter().cloned().fold(f32::INFINITY, f32::min))
        .collect();
    let maximin_row = (0..rows).max_by(|&a, &b| row_min[a].total_cmp(&row_min[b])).unwrap();
    let v_lower = row_min[maximin_row];
    let col_max: Vec<f32> = (0..cols)
        .map(|j| (0..rows).map(|i| m[i][j]).fold(f32::NEG_INFINITY, f32::max))
        .collect();
    let v_upper = col_max.iter().cloned().fold(f32::INFINITY, f32::min);
    if v_upper - v_lower < SADDLE_EPS {
        // 鞍点でも maximin がタイの行が複数あるとき、任意の1行を選ぶと弱支配される行
        // (均衡経路上は同値だが相手の逸脱にだけ損) を掴みうる。trembling-hand の一次近似
        // として、タイ行の中から「全列平均ペイオフ最大」の行を選ぶ (相手の微小な逸脱に強い)。
        let tied: Vec<usize> = (0..rows)
            .filter(|&i| v_lower - row_min[i] < SADDLE_EPS)
            .collect();
        let best = *tied
            .iter()
            .max_by(|&&a, &&b| {
                let sa: f32 = m[a].iter().sum();
                let sb: f32 = m[b].iter().sum();
                sa.total_cmp(&sb)
            })
            .unwrap();
        let mut s = vec![0.0; rows];
        s[best] = 1.0;
        return (0.5 * (v_lower + v_upper), s);
    }
    if rows == 2 && cols == 2 {
        // 真の非鞍点 2×2 は内部混合を持ち denom != 0 (gap>EPS が保証)。
        let (a, b, c, d) = (m[0][0], m[0][1], m[1][0], m[1][1]);
        let denom = a - b - c + d;
        let p = ((d - c) / denom).clamp(0.0, 1.0);
        let val = (a * d - b * c) / denom;
        return (val, vec![p, 1.0 - p]);
    }
    support_enumeration(m, rows, cols).unwrap_or_else(|| fictitious_play(m, rows, cols))
}

/// 4×4 以下を主対象にした support enumeration。支持集合上で無差別条件を解き、
/// 支持外を含む minmax 不等式を満たす候補だけを採用する。
fn support_enumeration(m: &[Vec<f32>], rows: usize, cols: usize) -> Option<(f32, Vec<f32>)> {
    let max_k = rows.min(cols);
    for k in 2..=max_k {
        for rm in 1usize..(1usize << rows) {
            if rm.count_ones() as usize != k { continue; }
            let ri: Vec<usize> = (0..rows).filter(|i| rm & (1 << i) != 0).collect();
            for cm in 1usize..(1usize << cols) {
                if cm.count_ones() as usize != k { continue; }
                let cj: Vec<usize> = (0..cols).filter(|j| cm & (1 << j) != 0).collect();
                // M[I,J]^T p = v1, sum(p)=1。
                let mut ap = vec![vec![0.0f64; k + 2]; k + 1];
                for (eq, &j) in cj.iter().enumerate() {
                    for (x, &i) in ri.iter().enumerate() { ap[eq][x] = m[i][j] as f64; }
                    ap[eq][k] = -1.0;
                }
                for x in 0..k { ap[k][x] = 1.0; }
                ap[k][k + 1] = 1.0;
                let Some(xp) = solve_linear(ap) else { continue };
                // M[I,J] q = v1, sum(q)=1。
                let mut aq = vec![vec![0.0f64; k + 2]; k + 1];
                for (eq, &i) in ri.iter().enumerate() {
                    for (x, &j) in cj.iter().enumerate() { aq[eq][x] = m[i][j] as f64; }
                    aq[eq][k] = -1.0;
                }
                for x in 0..k { aq[k][x] = 1.0; }
                aq[k][k + 1] = 1.0;
                let Some(xq) = solve_linear(aq) else { continue };
                let (vp, vq) = (xp[k], xq[k]);
                const EPS: f64 = 2e-6;
                if xp[..k].iter().any(|&x| x < -EPS)
                    || xq[..k].iter().any(|&x| x < -EPS)
                    || (vp - vq).abs() > EPS
                { continue; }
                let row_pay = |i: usize| -> f64 {
                    cj.iter().enumerate().map(|(x, &j)| m[i][j] as f64 * xq[x]).sum()
                };
                let col_pay = |j: usize| -> f64 {
                    ri.iter().enumerate().map(|(x, &i)| xp[x] * m[i][j] as f64).sum()
                };
                if (0..rows).any(|i| row_pay(i) > vp + EPS)
                    || (0..cols).any(|j| col_pay(j) < vp - EPS)
                { continue; }
                let mut strat = vec![0.0f32; rows];
                for (x, &i) in ri.iter().enumerate() { strat[i] = xp[x].max(0.0) as f32; }
                let sum: f32 = strat.iter().sum();
                if sum <= 0.0 { continue; }
                for p in &mut strat { *p /= sum; }
                return Some((vp as f32, strat));
            }
        }
    }
    None
}

fn solve_linear(mut a: Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let n = a.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col].abs() < 1e-10 { return None; }
        a.swap(col, pivot);
        let div = a[col][col];
        for j in col..=n { a[col][j] /= div; }
        for i in 0..n {
            if i == col { continue; }
            let f = a[i][col];
            for j in col..=n { a[i][j] -= f * a[col][j]; }
        }
    }
    Some((0..n).map(|i| a[i][n]).collect())
}

/// 一般ゼロ和の fictitious play (3c 以降の >2 行動用フォールバック)。
fn fictitious_play(m: &[Vec<f32>], rows: usize, cols: usize) -> (f32, Vec<f32>) {
    let iters = 1000;
    let mut row_cnt = vec![0.0f32; rows];
    let mut col_cnt = vec![0.0f32; cols];
    let mut col_hist = vec![0.0f32; cols];
    let mut row_hist = vec![0.0f32; rows];
    for _ in 0..iters {
        // 行は列の経験分布へ best response。
        let mut bi = 0;
        let mut bv = f32::NEG_INFINITY;
        for i in 0..rows {
            let val: f32 = (0..cols).map(|j| m[i][j] * col_hist[j]).sum();
            if val > bv { bv = val; bi = i; }
        }
        row_cnt[bi] += 1.0;
        for i in 0..rows { row_hist[i] = row_cnt[i]; }
        let rs: f32 = row_hist.iter().sum();
        for i in 0..rows { row_hist[i] /= rs; }
        // 列は行の経験分布へ best response (最小化)。
        let mut cj = 0;
        let mut cvv = f32::INFINITY;
        for j in 0..cols {
            let val: f32 = (0..rows).map(|i| m[i][j] * row_hist[i]).sum();
            if val < cvv { cvv = val; cj = j; }
        }
        col_cnt[cj] += 1.0;
        for j in 0..cols { col_hist[j] = col_cnt[j]; }
        let cs: f32 = col_hist.iter().sum();
        for j in 0..cols { col_hist[j] /= cs; }
    }
    let rs: f32 = row_cnt.iter().sum();
    let strat: Vec<f32> = row_cnt.iter().map(|c| c / rs).collect();
    let val: f32 = (0..rows).flat_map(|i| (0..cols).map(move |j| (i, j)))
        .map(|(i, j)| m[i][j] * strat[i] * col_hist[j]).sum();
    (val, strat)
}

/// 1 状態を解く: (ゲーム値 V, P(交代))。
pub(crate) fn solve_state(
    k: u64,
    v: &[f32],
    tmpl: &[poke_sho_rust::battle::BattleState; 2],
    cfg: &TransCfg,
    discount: f32,
) -> (f32, f32) {
    let (m, a1) = build_matrix(k, v, tmpl, cfg, discount);
    let (val, strat) = solve_zero_sum(&m);
    let pswitch: f32 = a1.iter().zip(strat.iter())
        .filter(|(c, _)| matches!(c, Choice::Switch(_)))
        .map(|(_, s)| *s)
        .sum();
    (val, pswitch)
}

/// 状態 k のペイオフ行列 (P1=行, 最大化) と P1 の合法手を返す。
pub(crate) fn build_matrix(
    k: u64,
    v: &[f32],
    tmpl: &[poke_sho_rust::battle::BattleState; 2],
    cfg: &TransCfg,
    discount: f32,
) -> (Vec<Vec<f32>>, Vec<Choice>) {
    let d = Dims::decompose(k, cfg.h);
    let st = fill(&tmpl[d.ai_team as usize], &d, cfg.h);
    let a1 = legal(&st, Player::P1);
    let a2 = legal(&st, Player::P2);
    // Shapley 割引確率ゲーム: 継続は γ·V(s')、終端は素の勝敗値。draw 注入は入れない
    // (入れると「無限交代でドロー(0.5)」という退化均衡が生まれ、優劣が伝播せず全状態が
    // 0.5・純戦略に潰れる)。γ<1 が無限交代を無価値にし、局面を決着へ向かわせる。対称な
    // 自己鏡像局面は対称性から厳密に 0.5 になる。
    // discount<1 のとき「毎手 (1-γ) の確率でゲームが打ち切られ、その時点のタイブレーク値を
    // 支払う」幾何打ち切りモデル: 継続 = γ·V(s') + (1-γ)·tiebreak(s')。定数 0.5 を注入する旧
    // draw 注入と違い、注入値が状態の優劣 (残 HP) を反映するので stall は不利側にとって損に
    // なり、退化均衡 (相互交代→0.5) が生じない。γ=1 なら素の undiscounted。
    let cont = |o: &Outcome| -> f32 {
        match *o {
            Outcome::Index(k2) => {
                let tb = crate::nash_vi_game::tiebreak(&Dims::decompose(k2, cfg.h), cfg.h);
                discount * v[k2 as usize] + (1.0 - discount) * tb
            }
            Outcome::Terminal(t) => t,
        }
    };
    let mut m: Vec<Vec<f32>> = Vec::with_capacity(a1.len());
    for &c1 in &a1 {
        let mut row = Vec::with_capacity(a2.len());
        for &c2 in &a2 {
            let tr = transition(&st, d.ai_team, c1, c2, cfg);
            let val: f32 = tr.iter().map(|(o, w)| w * cont(o)).sum();
            row.push(val);
        }
        m.push(row);
    }
    (m, a1)
}

/// VI 本体: 収束した V と有効 index 列を返す。
fn run_vi(
    h: u64,
    cfg: &TransCfg,
    tmpl: &[poke_sho_rust::battle::BattleState; 2],
    discount: f32,
    tol: f32,
    max_iters: u32,
    verbose: bool,
) -> (Vec<f32>, Vec<u64>) {
    let total = (8 * h * h * h * h) as usize;
    let valid: Vec<u64> = (0..total as u64)
        .filter(|&k| Dims::decompose(k, h).active_alive())
        .collect();
    if verbose {
        println!("[nash-vi] H={h} total={total} valid={}", valid.len());
    }
    let mut v = vec![0.5f32; total];
    let mut it = 0u32;
    loop {
        let newvals: Vec<f32> = valid
            .par_iter()
            .map(|&k| solve_state(k, &v, tmpl, cfg, discount).0)
            .collect();
        let mut delta = 0.0f32;
        for (i, &k) in valid.iter().enumerate() {
            let nv = newvals[i];
            delta = delta.max((nv - v[k as usize]).abs());
            v[k as usize] = nv;
        }
        it += 1;
        if verbose && (it % 10 == 0 || delta < tol) {
            println!("[nash-vi] iter {it}: max|ΔV|={delta:.2e}");
        }
        if delta < tol || it >= max_iters {
            break;
        }
    }
    (v, valid)
}

/// デバッグ: 収束後、指定 dense index の行列/解/value gap をダンプする。
/// 返りは各 k につき [rows, cols, m..(row-major).., val, pswitch, gap]。
#[pyfunction]
#[pyo3(signature = (stage, hp_buckets, ks, crit=true, randomize=true, discount=0.99,
                    tol=1e-5, max_iters=3000))]
#[allow(clippy::too_many_arguments)]
pub fn debug_nash_matrices(
    stage: &str,
    hp_buckets: u64,
    ks: Vec<u64>,
    crit: bool,
    randomize: bool,
    discount: f32,
    tol: f32,
    max_iters: u32,
) -> PyResult<Vec<Vec<f32>>> {
    let stage = Stage::from_short_name(stage)
        .ok_or_else(|| PyValueError::new_err("unknown stage"))?;
    let h = hp_buckets;
    let cfg = TransCfg { h, crit, randomize };
    let tmpl = templates(stage);
    let (v, _) = run_vi(h, &cfg, &tmpl, discount, tol, max_iters, false);
    let mut out = Vec::new();
    for k in ks {
        let (m, a1) = build_matrix(k, &v, &tmpl, &cfg, discount);
        let rows = m.len();
        let cols = if rows > 0 { m[0].len() } else { 0 };
        let (val, strat) = solve_zero_sum(&m);
        let pswitch: f32 = a1.iter().zip(strat.iter())
            .filter(|(c, _)| matches!(c, Choice::Switch(_)))
            .map(|(_, s)| *s)
            .sum();
        // value gap = minimax - maximin (0 なら鞍点=純)。
        let row_min: Vec<f32> = m.iter()
            .map(|r| r.iter().cloned().fold(f32::INFINITY, f32::min)).collect();
        let v_lower = row_min.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let v_upper = (0..cols)
            .map(|j| (0..rows).map(|i| m[i][j]).fold(f32::NEG_INFINITY, f32::max))
            .fold(f32::INFINITY, f32::min);
        let mut rec = vec![rows as f32, cols as f32];
        for row in &m { rec.extend_from_slice(row); }
        rec.push(val);
        rec.push(pswitch);
        rec.push(v_upper - v_lower);
        out.push(rec);
    }
    Ok(out)
}

/// Nash 値反復を回して web 形式の (P(交代)*1000, V*1000) を u16 で返す。
/// 無効 (アクティブ瀕死) index は SENTINEL。長さは 8·H^4。
#[pyfunction]
#[pyo3(signature = (stage, hp_buckets, crit=true, randomize=true, discount=0.99,
                    tol=1e-5, max_iters=2000, verbose=true))]
#[allow(clippy::too_many_arguments)]
pub fn solve_nash_vi(
    stage: &str,
    hp_buckets: u64,
    crit: bool,
    randomize: bool,
    discount: f32,
    tol: f32,
    max_iters: u32,
    verbose: bool,
) -> PyResult<(Vec<u16>, Vec<u16>, Vec<u16>, f32, f32)> {
    let stage = Stage::from_short_name(stage)
        .ok_or_else(|| PyValueError::new_err("unknown stage"))?;
    if !stage.is_party() {
        return Err(PyValueError::new_err("nash VI needs a party stage (3b)"));
    }
    let h = hp_buckets;
    let total = (8 * h * h * h * h) as usize;
    let cfg = TransCfg { h, crit, randomize };
    let tmpl = templates(stage);
    if verbose {
        println!("[nash-vi] stage={stage:?}");
    }
    // フェーズ1: 割引 VI で戦略 σ (= P(交代)) を得る。割引が決着均衡を選ぶ。
    let (v_disc, valid) = run_vi(h, &cfg, &tmpl, discount, tol, max_iters, verbose);
    let mut pswitch_full = vec![0.0f32; total];
    let strat: Vec<(u64, f32)> = valid
        .par_iter()
        .map(|&k| (k, solve_state(k, &v_disc, &tmpl, &cfg, discount).1))
        .collect();
    for (k, ps) in strat {
        pswitch_full[k as usize] = ps;
    }

    // フェーズ2: σ の**割引なし**方策評価で真の勝率 V を得る (対称局面=0.5)。
    let v0 = vec![0.5f32; total];
    let v_true = policy_eval(&v0, &valid, &pswitch_full, &tmpl, &cfg, 1.0, tol, max_iters);

    // フェーズ3: P1 best-response で exploitability を測る (均衡なら V_br≈V_true)。
    let v_br = best_response(&v_true, &valid, &pswitch_full, &tmpl, &cfg, 1.0, tol, max_iters);
    let mut expl_max = 0.0f32;
    let mut expl_sum = 0.0f32;
    for &k in &valid {
        let g = (v_br[k as usize] - v_true[k as usize]).max(0.0);
        expl_max = expl_max.max(g);
        expl_sum += g;
    }
    let expl_mean = expl_sum / valid.len() as f32;
    if verbose {
        println!("[nash-vi] exploitability: mean={expl_mean:.4} max={expl_max:.4}");
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
    if verbose {
        println!("[nash-vi] done");
    }
    Ok((policy, value, br, expl_mean, expl_max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_sum_2x2_mixed() {
        // マッチングペニー的: [[1,0],[0,1]] → 値 0.5, 戦略 [0.5,0.5]。
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (v, s) = solve_zero_sum(&m);
        assert!((v - 0.5).abs() < 1e-4);
        assert!((s[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn zero_sum_saddle_pure() {
        // 支配のある行列は純戦略。行1が常に良い。
        let m = vec![vec![0.2, 0.3], vec![0.6, 0.7]];
        let (v, s) = solve_zero_sum(&m);
        assert!((v - 0.6).abs() < 1e-6);
        assert!((s[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_sum_3x3_full_support() {
        let m = vec![vec![0.5, 1.0, 0.0], vec![0.0, 0.5, 1.0], vec![1.0, 0.0, 0.5]];
        let (v, s) = solve_zero_sum(&m);
        assert!((v - 0.5).abs() < 1e-4);
        for p in s { assert!((p - 1.0 / 3.0).abs() < 1e-4); }
    }

    #[test]
    fn zero_sum_4x4_can_ignore_dominated_action() {
        let m = vec![
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.2, 0.2, 0.2, 0.2],
            vec![0.1, 0.1, 0.1, 0.1],
        ];
        let (v, s) = solve_zero_sum(&m);
        assert!((v - 0.5).abs() < 1e-4);
        assert!((s[0] - 0.5).abs() < 1e-4 && (s[1] - 0.5).abs() < 1e-4);
    }
}
