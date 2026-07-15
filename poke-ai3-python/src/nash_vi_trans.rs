//! 厳密遷移列挙: `(s, a1, a2)` → 次状態の確率分布。
//!
//! 手番順は公平コイン 1/2、急所は 1/24、ダメージロールは 85..=100 の一様16値
//! (すべて独立)。ダメージ適用は既存の決定論 API `execute_action(state, attacker,
//! move, crit, roll)` を crit/roll 注入で再利用し、シミュレータのターン解決規則
//! (交代はフェーズ1で先行・技は速度順・KO された側は撃てない・ターン末に強制交代) を
//! そのまま辿る。控えは 3b では一意なので強制交代に決定は不要。

use std::collections::HashMap;

use poke_sho_rust::battle::{BattleState, Choice, Player};
use poke_sho_rust::damage::{DamageInput, calc_damage};
use poke_sho_rust::scenario::MoveId;

use crate::nash_vi_game::{Outcome, classify};

/// アロケーション無しでダメージを適用する (`execute_action` から event 生成を除いた版)。
/// 巨大な列挙で event Vec を毎回確保しないためのホットパス。ダメージ式は同一。
fn apply_move(st: &mut BattleState, attacker: Player, mv: MoveId, crit: bool, roll: u8) {
    let atk = st.pokemon(attacker);
    let target = attacker.opponent();
    let def = st.pokemon(target);
    let data = mv.data();
    let dmg = calc_damage(&DamageInput {
        level: atk.level,
        attacker: &atk.stats,
        attacker_types: atk.types,
        defender: &def.stats,
        defender_types: def.types,
        mv: &data,
        crit,
        roll,
    });
    let party = &mut st.parties[target.index()];
    let a = party.active;
    let mon = &mut party.members[a];
    let actual = dmg.min(mon.hp);
    mon.hp -= actual;
}

pub struct TransCfg {
    pub h: u64,
    pub crit: bool,
    pub randomize: bool,
}

/// 1 攻撃の (crit, roll, 確率) 分布。
fn attack_outcomes(cfg: &TransCfg) -> Vec<(bool, u8, f32)> {
    let crits: Vec<(bool, f32)> = if cfg.crit {
        vec![(true, 1.0 / 24.0), (false, 23.0 / 24.0)]
    } else {
        vec![(false, 1.0)]
    };
    let rolls: Vec<(u8, f32)> = if cfg.randomize {
        (85u8..=100).map(|r| (r, 1.0 / 16.0)).collect()
    } else {
        vec![(100, 1.0)]
    };
    let mut out = Vec::with_capacity(crits.len() * rolls.len());
    for &(c, pc) in &crits {
        for &(r, pr) in &rolls {
            out.push((c, r, pc * pr));
        }
    }
    out
}

/// 場のアクティブが瀕死かつ控えが生存する側に、一意な控えを出す (強制交代)。
fn resolve_forced_switch(st: &mut BattleState) {
    for side in 0..2 {
        let party = &mut st.parties[side];
        if party.active_mon().hp <= 0 {
            // 生存している別メンバー (3b は 2 体なので一意) を探す。
            for i in 0..party.len {
                if i != party.active && party.members[i].hp > 0 {
                    party.active = i;
                    break;
                }
            }
        }
    }
}

/// `(s, a1, a2)` の遷移分布を列挙する。返り値は Outcome ごとの確率 (合計 1.0)。
pub fn transition(
    base: &BattleState,
    ai_team: u64,
    a1: Choice,
    a2: Choice,
    cfg: &TransCfg,
) -> Vec<(Outcome, f32)> {
    // フェーズ1: 交代 (両者・順序非依存)。交代した側はこのターン技を撃たない。
    let mut post_switch = *base;
    if let Choice::Switch(idx) = a1 {
        post_switch.parties[0].active = idx;
    }
    if let Choice::Switch(idx) = a2 {
        post_switch.parties[1].active = idx;
    }
    let move_of = |c: Choice| match c {
        Choice::Move(m) => Some(m),
        Choice::Switch(_) => None,
    };
    let m1 = move_of(a1);
    let m2 = move_of(a2);

    let atk = attack_outcomes(cfg);
    // 終端値は連続 index と別扱いするため、index は u64 キー・終端は f32 キーで集計。
    let mut idx_acc: HashMap<u64, f32> = HashMap::new();
    let mut term_acc: HashMap<u32, f32> = HashMap::new(); // value*1e6 を鍵に量子化

    // 手番順コイン (P1 先手 / P2 先手)、各 0.5。
    for &first in &[Player::P1, Player::P2] {
        let order = [first, first.opponent()];
        // このターン技を撃つ手番を順序どおりに (交代側は撃たない)。
        let seq: Vec<(Player, poke_sho_rust::scenario::MoveId)> = order
            .iter()
            .filter_map(|&p| {
                let mv = match p {
                    Player::P1 => m1,
                    Player::P2 => m2,
                };
                mv.map(|m| (p, m))
            })
            .collect();

        // seq 上で crit/roll を直積列挙。KO された手番は撃てない (rng 消費なし)。
        let mut branches: Vec<(BattleState, f32)> = vec![(post_switch, 0.5)];
        for &(p, mv) in &seq {
            let mut next: Vec<(BattleState, f32)> = Vec::new();
            for (st, w) in branches.drain(..) {
                if st.parties[p.index()].active_mon().hp <= 0 {
                    // 既に瀕死 → 撃てない。単一分岐で通過。
                    next.push((st, w));
                    continue;
                }
                for &(crit, roll, wa) in &atk {
                    let mut c = st;
                    apply_move(&mut c, p, mv, crit, roll);
                    next.push((c, w * wa));
                }
            }
            branches = next;
        }

        for (mut st, w) in branches {
            resolve_forced_switch(&mut st);
            match classify(&st, ai_team, cfg.h) {
                Outcome::Index(k) => *idx_acc.entry(k).or_insert(0.0) += w,
                Outcome::Terminal(v) => {
                    *term_acc.entry((v * 1.0e6).round() as u32).or_insert(0.0) += w
                }
            }
        }
    }

    let mut out: Vec<(Outcome, f32)> = Vec::with_capacity(idx_acc.len() + term_acc.len());
    for (k, w) in idx_acc {
        out.push((Outcome::Index(k), w));
    }
    for (vq, w) in term_acc {
        out.push((Outcome::Terminal(vq as f32 / 1.0e6), w));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nash_vi_game::{Dims, build_state, legal};
    use poke_sho_rust::scenario::Stage;

    fn cfg() -> TransCfg {
        TransCfg { h: 26, crit: true, randomize: true }
    }

    #[test]
    fn probs_sum_to_one() {
        let h = 26;
        let d = Dims { ai_team: 0, ai_active: 0, ai_hp_c: 25, ai_hp_g: 25,
                       opp_active: 0, opp_hp_c: 25, opp_hp_g: 25 };
        let st = build_state(&d, h, Stage::Stage3b);
        let a1 = legal(&st, Player::P1);
        let a2 = legal(&st, Player::P2);
        for &c1 in &a1 {
            for &c2 in &a2 {
                let tr = transition(&st, d.ai_team, c1, c2, &cfg());
                let s: f32 = tr.iter().map(|(_, w)| *w).sum();
                assert!((s - 1.0).abs() < 1e-4, "sum={s} for {c1:?} {c2:?}");
            }
        }
    }

    #[test]
    fn low_hp_move_can_ko_to_terminal() {
        // 相手 active を瀕死寸前 (bucket 1) にして攻撃すれば終局分岐が出る。
        let h = 26;
        let d = Dims { ai_team: 0, ai_active: 0, ai_hp_c: 25, ai_hp_g: 0,
                       opp_active: 0, opp_hp_c: 1, opp_hp_g: 0 };
        let st = build_state(&d, h, Stage::Stage3b);
        // P1 は技、P2 も技。P1 が当てれば P2 は全滅 (控え Goodra も瀕死) → 終局。
        let m1 = legal(&st, Player::P1).into_iter().find(|c| matches!(c, Choice::Move(_))).unwrap();
        let m2 = legal(&st, Player::P2).into_iter().find(|c| matches!(c, Choice::Move(_))).unwrap();
        let tr = transition(&st, d.ai_team, m1, m2, &cfg());
        let has_term = tr.iter().any(|(o, w)| matches!(o, Outcome::Terminal(_)) && *w > 0.0);
        assert!(has_term, "expected a terminal (KO) branch");
    }
}
