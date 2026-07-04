//! 評価用の固定ルールベース方策。
//!
//! ユーザーが「最善手」とみなす決定論方策を実装する:
//! - active が相手 active に super-effective (SE) を撃てるなら、その SE 技で攻撃。
//! - 撃てないなら、相手に SE を撃てる相方へ交代する。
//! - どちらも不可能 (相方が SE を持たない異常系/強制交代) なら、合法手の先頭へ縮退。
//!
//! lookahead も推論も介さず、真の `BattleState` だけから着手を決める (Local 専用)。
//! 学習はせず、学習済み AI の交代タイミングの妥当性を固定ベースラインで測るために使う。

use poke_env_rust::observation::{BattleState, Choice, MoveId, Player, SpeciesId};

/// 相手 active 種族に super-effective (2倍) が通る技 (フォルム厳密)。
/// Python 側 `diagnostics._se_set_for` と同一の対応:
/// - Cloyster (Water/Ice): Shock Wave(電→水) / FightSpe60(闘→氷)。
/// - Goodra-Hisui (Steel/Dragon): Bulldoze(地→鋼) / FairyPhy60(妖→竜)。
/// - 原種 Goodra (Dragon): FairyPhy60(妖→竜) のみ (Bulldoze は等倍)。
/// 3a/3b は前者2技、3c は後者2技を使うが、種族で集合を返すので stage を問わない。
fn se_moves_for(opp: SpeciesId) -> &'static [MoveId] {
    match opp {
        SpeciesId::Cloyster => &[MoveId::ShockWave, MoveId::FightSpe60],
        SpeciesId::GoodraHisui => &[MoveId::Bulldoze, MoveId::FairyPhy60],
        SpeciesId::Goodra => &[MoveId::FairyPhy60],
    }
}

/// 固定ルールに従って着手を決める。通常ターン・強制交代の両方を同じ判定で扱う
/// (強制交代では合法手が交代のみになるので、自然に SE 相方への交代が選ばれる)。
pub fn rule_choice(state: &BattleState, player: Player) -> Choice {
    let legal = state.legal_choices(player);
    let opp_species = state.party(player.opponent()).active_mon().species_id;
    let se = se_moves_for(opp_species);

    // 1. active が SE を撃てるなら攻撃。
    for c in &legal {
        if let Choice::Move(m) = c {
            if se.contains(m) {
                return *c;
            }
        }
    }
    // 2. SE を撃てる相方へ交代。
    let party = state.party(player);
    for c in &legal {
        if let Choice::Switch(abs) = c {
            if se.iter().any(|m| party.members[*abs].moves[m.index()]) {
                return *c;
            }
        }
    }
    // 3. 縮退: 合法な技があればそれ、無ければ合法手の先頭 (交代)。
    for c in &legal {
        if let Choice::Move(_) = c {
            return *c;
        }
    }
    legal[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_env_rust::observation::{BattleState, Stage, TeamId};

    /// Stage3b の初期状態を (P1先発, P2先発) で作る。両側 {Cloyster, Goodra-Hisui}。
    fn state(p1: SpeciesId, p2: SpeciesId) -> BattleState {
        BattleState::new_with_teams(
            Stage::Stage3b,
            (TeamId::Team1, p1.index()),
            (TeamId::Team1, p2.index()),
        )
    }

    #[test]
    fn se_attacks_when_active_can_hit_super_effective() {
        // P1=Cloyster(Shock Wave) vs P2=Cloyster: Shock Wave は Cloyster に SE → 攻撃。
        let s = state(SpeciesId::Cloyster, SpeciesId::Cloyster);
        assert_eq!(rule_choice(&s, Player::P1), Choice::Move(MoveId::ShockWave));
        // P1=Goodra(Bulldoze) vs P2=Goodra: Bulldoze は Goodra-Hisui に SE → 攻撃。
        let s = state(SpeciesId::GoodraHisui, SpeciesId::GoodraHisui);
        assert_eq!(rule_choice(&s, Player::P1), Choice::Move(MoveId::Bulldoze));
    }

    #[test]
    fn switches_to_se_partner_when_active_cannot_hit_se() {
        // P1=Cloyster(Shock Wave) vs P2=Goodra: Shock Wave は半減 → SE を撃てる相方
        // (Goodra/Bulldoze) へ交代。控えは 1 体なのでその交代手。
        let s = state(SpeciesId::Cloyster, SpeciesId::GoodraHisui);
        match rule_choice(&s, Player::P1) {
            Choice::Switch(abs) => {
                assert!(s.party(Player::P1).members[abs].moves[MoveId::Bulldoze.index()]);
            }
            other => panic!("expected switch to Bulldoze partner, got {other:?}"),
        }
        // P1=Goodra(Bulldoze) vs P2=Cloyster: 等倍 → SE を撃てる相方 (Cloyster/Shock Wave) へ。
        let s = state(SpeciesId::GoodraHisui, SpeciesId::Cloyster);
        match rule_choice(&s, Player::P1) {
            Choice::Switch(abs) => {
                assert!(s.party(Player::P1).members[abs].moves[MoveId::ShockWave.index()]);
            }
            other => panic!("expected switch to Shock Wave partner, got {other:?}"),
        }
    }

    /// Stage3c (対称対面) でも SE 技/相方交代が正しく選ばれる。
    #[test]
    fn stage3c_se_attack_and_switch() {
        // 宣言順: 0=Cloyster(FightSpe60 for team1), 1=Goodra(FairyPhy60 for team1)。
        let s = BattleState::new_with_teams(
            Stage::Stage3c,
            (TeamId::Team1, 0),
            (TeamId::Team1, 0),
        );
        // P1=Cloyster(FightSpe60) vs P2=Cloyster: FightSpe60 は Cloyster(Ice) に SE → 攻撃。
        assert_eq!(rule_choice(&s, Player::P1), Choice::Move(MoveId::FightSpe60));
        // P1=Cloyster(FightSpe60) vs P2=Goodra: 等倍 → SE を撃てる相方 (Goodra/FairyPhy60) へ交代。
        let s = BattleState::new_with_teams(
            Stage::Stage3c,
            (TeamId::Team1, 0),
            (TeamId::Team1, 1),
        );
        match rule_choice(&s, Player::P1) {
            Choice::Switch(abs) => {
                assert!(s.party(Player::P1).members[abs].moves[MoveId::FairyPhy60.index()]);
            }
            other => panic!("expected switch to FairyPhy60 partner, got {other:?}"),
        }
    }
}
