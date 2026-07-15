//! 固定戦略 σ の**割引なし**評価と best-response。
//!
//! Nash 値反復 (割引あり) が選ぶ決着均衡の戦略 σ に対し、真の勝率 V (割引なし・
//! 対称局面=0.5) を σ の方策評価で求め、さらに P1 の best-response 値で exploitability
//! を測る。σ が混合していれば相互交代ループは確率<1 で必ず破れる (proper) ので、割引
//! なしでも一意に収束する。σ_P2 は対称性から `pswitch[swap(s)]` で引く。

use poke_sho_rust::battle::{BattleState, Choice, Player};

use crate::nash_vi_game::{Dims, Outcome, fill, legal};
use crate::nash_vi_trans::{TransCfg, transition};

/// 合法手 `a` と P(交代) から着手分布を作る (技先頭・交代末尾、交代質量を交代手に等分)。
fn strat(a: &[Choice], pswitch: f32) -> Vec<f32> {
    let n_sw = a.iter().filter(|c| matches!(c, Choice::Switch(_))).count();
    if n_sw == 0 || a.len() == n_sw {
        return vec![1.0 / a.len() as f32; a.len()];
    }
    let n_mv = a.len() - n_sw;
    a.iter()
        .map(|c| if matches!(c, Choice::Switch(_)) {
            pswitch / n_sw as f32
        } else {
            (1.0 - pswitch) / n_mv as f32
        })
        .collect()
}

/// 状態 k の期待値行列 `M[i][j] = E_遷移[ term? t : cont(v[s']) ]` と両者合法手。
/// discount<1 は幾何打ち切りゲーム: 継続 = γ·V(s') + (1-γ)·tiebreak(s') (solve と同じ作用素)。
fn expected_matrix(
    k: u64,
    v: &[f32],
    tmpl: &[BattleState; 2],
    cfg: &TransCfg,
    discount: f32,
) -> (Vec<Vec<f32>>, Vec<Choice>, Vec<Choice>) {
    let d = Dims::decompose(k, cfg.h);
    let st = fill(&tmpl[d.ai_team as usize], &d, cfg.h);
    let a1 = legal(&st, Player::P1);
    let a2 = legal(&st, Player::P2);
    let cont = |o: &Outcome| -> f32 {
        match *o {
            Outcome::Index(k2) => {
                let tb = crate::nash_vi_game::tiebreak(&Dims::decompose(k2, cfg.h), cfg.h);
                discount * v[k2 as usize] + (1.0 - discount) * tb
            }
            Outcome::Terminal(t) => t,
        }
    };
    let mut m = Vec::with_capacity(a1.len());
    for &c1 in &a1 {
        let mut row = Vec::with_capacity(a2.len());
        for &c2 in &a2 {
            let tr = transition(&st, d.ai_team, c1, c2, cfg);
            row.push(tr.iter().map(|(o, w)| w * cont(o)).sum::<f32>());
        }
        m.push(row);
    }
    (m, a1, a2)
}

/// 固定戦略 σ (pswitch_full: P1 視点の全 index) の割引なし勝率 V を反復で求める。
/// P2 の σ は `pswitch_full[swap(s)]`。
pub fn policy_eval(
    v_init: &[f32],
    valid: &[u64],
    pswitch_full: &[f32],
    tmpl: &[BattleState; 2],
    cfg: &TransCfg,
    discount: f32,
    tol: f32,
    max_iters: u32,
) -> Vec<f32> {
    use rayon::prelude::*;
    let mut v = v_init.to_vec();
    for _ in 0..max_iters {
        let newv: Vec<f32> = valid
            .par_iter()
            .map(|&k| {
                let (m, a1, a2) = expected_matrix(k, &v, tmpl, cfg, discount);
                let d = Dims::decompose(k, cfg.h);
                let s1 = strat(&a1, pswitch_full[k as usize]);
                let s2 = strat(&a2, pswitch_full[d.swap().compose(cfg.h) as usize]);
                let mut acc = 0.0;
                for (i, &si) in s1.iter().enumerate() {
                    for (j, &sj) in s2.iter().enumerate() {
                        acc += si * sj * m[i][j];
                    }
                }
                acc
            })
            .collect();
        let mut delta = 0.0f32;
        for (idx, &k) in valid.iter().enumerate() {
            delta = delta.max((newv[idx] - v[k as usize]).abs());
            v[k as usize] = newv[idx];
        }
        if delta < tol {
            break;
        }
    }
    v
}

/// P1 が best-response (P2 は固定 σ) したときの割引なし勝率 V を求める。
/// 収束後の各 start 状態の値が exploitability の測定に使える (均衡なら ≈0.5)。
pub fn best_response(
    v_init: &[f32],
    valid: &[u64],
    pswitch_full: &[f32],
    tmpl: &[BattleState; 2],
    cfg: &TransCfg,
    discount: f32,
    tol: f32,
    max_iters: u32,
) -> Vec<f32> {
    use rayon::prelude::*;
    let mut v = v_init.to_vec();
    for _ in 0..max_iters {
        let newv: Vec<f32> = valid
            .par_iter()
            .map(|&k| {
                let (m, _a1, a2) = expected_matrix(k, &v, tmpl, cfg, discount);
                let d = Dims::decompose(k, cfg.h);
                let s2 = strat(&a2, pswitch_full[d.swap().compose(cfg.h) as usize]);
                // P1 は各手 i の期待値 Σ_j σ2[j] M[i][j] の最大を取る。
                m.iter()
                    .map(|row| row.iter().zip(&s2).map(|(mij, sj)| mij * sj).sum::<f32>())
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        let mut delta = 0.0f32;
        for (idx, &k) in valid.iter().enumerate() {
            delta = delta.max((newv[idx] - v[k as usize]).abs());
            v[k as usize] = newv[idx];
        }
        if delta < tol {
            break;
        }
    }
    v
}
