//! バトルの状態モデルと合法手。
//!
//! 状態 (`BattleState` / `Choice` / `Player`) と合法手 (`legal_choices`) を持つ。
//! 各サイドは [`Party`] を持ち、1v1 ステージはメンバー 1 体・交代不可の退化ケースとして
//! 同じコードパスで扱う。ターン解決ロジックは `turn` モジュール、個体・パーティの型は
//! `party` モジュール、種族や技の定義は `scenario` モジュールにある。

use crate::party::Party;
pub use crate::party::{MAX_PARTY, PokemonState};
pub use crate::turn::{TurnResult, apply_forced_switches, apply_turn, execute_action};
use crate::scenario::MoveId;

/// 最大ターン数の既定値。`BattleState::max_turns` の初期値として使う。これを超えると
/// 無条件引き分け (`winner()=None`) で終局する。交代を撃ち合うと理論上は無限戦になり得る
/// ため、その上限としても機能する。正常な対戦は十数ターンで決着するので十分な余裕がある。
/// `max_turns = 0` にすると上限なし (引き分けによる打ち切りをしない)。
pub const MAX_TURNS: u32 = 100;

/// バトルのプレイヤー。
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
pub enum Player {
    P1,
    P2,
}

use serde::{Deserialize, Serialize};

impl Player {
    pub fn index(self) -> usize {
        match self {
            Player::P1 => 0,
            Player::P2 => 1,
        }
    }

    pub fn opponent(self) -> Player {
        match self {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1,
        }
    }
}

/// A player's choice for a turn: use a move, or switch to a party member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// 場の個体で技を撃つ。
    Move(MoveId),
    /// 指定したパーティ index の控えに交代する。
    Switch(usize),
}

/// バトル状態。両側のパーティはマッチごとに独立に決まる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BattleState {
    pub parties: [Party; 2],
    pub turn: u32,
    /// 瀕死により強制交代待ちのサイド。`true` の側は次に交代手だけが合法。
    pub forced_switch: [bool; 2],
    /// ターン上限 (到達で引き分け終局)。`0` は上限なし (打ち切りしない)。既定は `MAX_TURNS`。
    pub max_turns: u32,
}

impl BattleState {
    pub fn party(&self, player: Player) -> &Party {
        &self.parties[player.index()]
    }

    /// 場に出ている個体 (値で返す。`PokemonState` は `Copy`)。
    pub fn pokemon(&self, player: Player) -> PokemonState {
        *self.parties[player.index()].active_mon()
    }

    pub fn hp(&self, player: Player) -> i32 {
        self.parties[player.index()].active_mon().hp
    }

    /// 場の個体が瀕死か (敗北とは限らない。控えがいれば交代待ち)。
    pub fn is_fainted(&self, player: Player) -> bool {
        self.parties[player.index()].active_mon().hp <= 0
    }

    /// `player` がパーティ全滅で敗北したか。
    pub fn is_lost(&self, player: Player) -> bool {
        self.parties[player.index()].all_fainted()
    }

    pub fn winner(&self) -> Option<Player> {
        match (self.is_lost(Player::P1), self.is_lost(Player::P2)) {
            (false, true) => Some(Player::P1),
            (true, false) => Some(Player::P2),
            _ => None,
        }
    }

    /// バトルが終局したか。全滅決着に加え、`max_turns` 到達 (引き分け) も終局扱い。
    /// `max_turns == 0` のときは上限なし (全滅決着だけで終局)。
    pub fn is_done(&self) -> bool {
        self.is_lost(Player::P1)
            || self.is_lost(Player::P2)
            || (self.max_turns != 0 && self.turn >= self.max_turns)
    }

    /// ターン上限を差し替えた状態を返す (`0` で上限なし)。生成後に web/CLI 側で設定する用。
    pub fn with_max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// ターン上限による引き分けで終局したか (勝者なしの終局)。
    pub fn is_draw(&self) -> bool {
        self.is_done() && self.winner().is_none()
    }

    /// `player` が瀕死後の強制交代待ちか (交代手だけが合法な決定点)。
    /// local_showdown / lookahead が通常ターンと強制交代サブフェーズを
    /// 同じ判定で見分けるための共通アクセサ。
    pub fn needs_forced_switch(&self, player: Player) -> bool {
        self.forced_switch[player.index()]
    }

    /// いずれかの側が強制交代待ちか (この後 `apply_forced_switches` が要る)。
    pub fn any_forced_switch(&self) -> bool {
        self.forced_switch.iter().any(|&f| f)
    }

    pub fn legal_choices(&self, player: Player) -> Vec<Choice> {
        assert!(!self.is_done(), "legal_choices: called on a finished battle");

        let party = self.party(player);
        // 強制交代待ち: 控えへの交代手だけが合法 (技は撃てない)。
        if self.forced_switch[player.index()] {
            return party.switch_targets().map(Choice::Switch).collect();
        }
        assert!(
            party.active_mon().hp > 0,
            "legal_choices: active mon is fainted but forced_switch flag is not set"
        );
        let known = party.active_mon().moves;
        let mut choices: Vec<Choice> = MoveId::ALL
            .into_iter()
            .filter(|mv| known[mv.index()])
            .map(Choice::Move)
            .collect();
        choices.extend(party.switch_targets().map(Choice::Switch));
        // TODO: PP切れで手がなくなった場合は「わるあがき」を返すべきだが未実装。
        //       現状は発生しないので assert で検知する。
        assert!(!choices.is_empty(), "legal_choices: no legal choices available");
        choices
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::battle_rng::MaxRoll;
    use crate::event::Event;
    use crate::scenario::{SpeciesId, Stage, TeamId};

    /// 旧・単一チーム相当 (両側 Team1) で先発 Cloyster ミラーの Stage3b 状態。
    fn stage3b_cloyster_mirror() -> BattleState {
        BattleState::new_with_teams(
            Stage::Stage3b,
            (TeamId::Team1, SpeciesId::Cloyster.index()),
            (TeamId::Team1, SpeciesId::Cloyster.index()),
        )
    }

    fn turn(state: BattleState, p1: MoveId, p2: MoveId, first: Player) -> TurnResult {
        apply_turn(
            state,
            Choice::Move(p1),
            Choice::Move(p2),
            first,
            &mut MaxRoll,
        )
    }

    #[test]
    fn stage3a_legal_moves_are_four_and_exploit_weakness() {
        // Stage 3a は 4 技。Cloyster (Water/Ice) 対面では Shock Wave (でんき) が、
        // Goodra-Hisui (Steel/Dragon) 対面では Bulldoze (じめん) が弱点を突く。
        let state = BattleState::new(Stage::Stage3a, SpeciesId::GoodraHisui, SpeciesId::Cloyster);
        assert_eq!(state.legal_choices(Player::P1).len(), 4);
        // 攻撃側 Goodra-Hisui が防御側 Cloyster に撃つ: Shock Wave > Dark Pulse(等倍同分類)。
        let shock = turn(state, MoveId::ShockWave, MoveId::Crunch, Player::P1).state.hp(Player::P2);
        let pulse = turn(state, MoveId::DarkPulse, MoveId::Crunch, Player::P1).state.hp(Player::P2);
        assert!(shock < pulse, "Shock Wave should out-damage Dark Pulse on Cloyster (shock={shock}, pulse={pulse})");
        // 逆: 防御側 Goodra-Hisui へ Bulldoze (物理じめん 2倍) > Crunch (物理あく 等倍)。
        let state2 = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::GoodraHisui);
        let bull = turn(state2, MoveId::Bulldoze, MoveId::Crunch, Player::P1).state.hp(Player::P2);
        let crunch = turn(state2, MoveId::Crunch, MoveId::Crunch, Player::P1).state.hp(Player::P2);
        assert!(bull < crunch, "Bulldoze should out-damage Crunch on Goodra-Hisui (bull={bull}, crunch={crunch})");
    }

    #[test]
    fn dark_pulse_hits_cloyster_harder_than_crunch() {
        // 攻撃側=Goodra-Hisui (A=117, C=117 同値)、防御側=Cloyster (B=222, D=94)。
        // Crunch は B=222 にぶつかり、Dark Pulse は D=94 に刺さる (両技ともあく・等倍)。
        let state = BattleState::new(Stage::Stage3a, SpeciesId::GoodraHisui, SpeciesId::Cloyster);
        let r_crunch = turn(state, MoveId::Crunch, MoveId::Crunch, Player::P1);
        let r_pulse = turn(state, MoveId::DarkPulse, MoveId::Crunch, Player::P1);
        let cloy_hp_after_crunch = r_crunch.state.hp(Player::P2);
        let cloy_hp_after_pulse = r_pulse.state.hp(Player::P2);
        assert!(
            cloy_hp_after_pulse < cloy_hp_after_crunch,
            "Dark Pulse should leave Cloyster lower than Crunch (got pulse={cloy_hp_after_pulse}, crunch={cloy_hp_after_crunch})"
        );
    }

    #[test]
    fn crunch_hits_goodra_harder_than_dark_pulse() {
        // 逆対称: 防御側=Goodra-Hisui (B=94, D=222)。
        let state = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::GoodraHisui);
        let r_crunch = turn(state, MoveId::Crunch, MoveId::Crunch, Player::P1);
        let r_pulse = turn(state, MoveId::DarkPulse, MoveId::Crunch, Player::P1);
        let go_hp_after_crunch = r_crunch.state.hp(Player::P2);
        let go_hp_after_pulse = r_pulse.state.hp(Player::P2);
        assert!(
            go_hp_after_crunch < go_hp_after_pulse,
            "Crunch should leave Goodra lower than Dark Pulse (got crunch={go_hp_after_crunch}, pulse={go_hp_after_pulse})"
        );
    }

    #[test]
    fn ko_ends_battle_and_emits_win() {
        let mut state = BattleState::new(Stage::Stage3a, SpeciesId::GoodraHisui, SpeciesId::Cloyster);
        state.parties[Player::P2.index()].active_mon_mut().hp = 1;
        let result = turn(state, MoveId::DarkPulse, MoveId::Crunch, Player::P1);
        assert_eq!(result.state.hp(Player::P2), 0);
        assert_eq!(result.state.winner(), Some(Player::P1));
        assert!(result.events.iter().any(|e| matches!(e, Event::Win { .. })));
    }

    #[test]
    fn finished_battle_does_not_change() {
        let mut state = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::GoodraHisui);
        state.parties[Player::P1.index()].active_mon_mut().hp = 0;
        let original = state;
        let result = turn(state, MoveId::Crunch, MoveId::Crunch, Player::P1);
        assert_eq!(result.state, original);
        assert!(result.events.is_empty());
    }

    #[test]
    fn stage3b_has_two_members_and_switch_choice() {
        // Stage3b: 両側 {Cloyster, Goodra-Hisui} の 2 体。先発は引数で選ぶ。
        let state = BattleState::new_with_teams(
            Stage::Stage3b,
            (TeamId::Team1, SpeciesId::Cloyster.index()),
            (TeamId::Team1, SpeciesId::GoodraHisui.index()),
        );
        assert_eq!(state.party(Player::P1).len, 2);
        // P1 の先発は Cloyster (Shock Wave のみ)。合法手 = 技1 + 交代1 = 2。
        let p1 = state.pokemon(Player::P1);
        assert_eq!(p1.name, "Cloyster");
        let choices = state.legal_choices(Player::P1);
        assert_eq!(choices.iter().filter(|c| matches!(c, Choice::Move(_))).count(), 1);
        assert_eq!(choices.iter().filter(|c| matches!(c, Choice::Switch(_))).count(), 1);
        // P2 の先発は Goodra-Hisui。
        assert_eq!(state.pokemon(Player::P2).name, "Goodra-Hisui");
    }

    #[test]
    fn switch_brings_bench_in_before_move_hits() {
        // P1 先発 Goodra を Cloyster(idx0) に交代。交代は相手の技より先なので、
        // P2 の技は交代後の Cloyster に当たる。
        let state = BattleState::new_with_teams(
            Stage::Stage3b,
            (TeamId::Team1, SpeciesId::GoodraHisui.index()),
            (TeamId::Team1, SpeciesId::Cloyster.index()),
        );
        assert_eq!(state.pokemon(Player::P1).name, "Goodra-Hisui");
        let result = apply_turn(
            state,
            Choice::Switch(SpeciesId::Cloyster.index()),
            Choice::Move(MoveId::ShockWave),
            Player::P2,
            &mut MaxRoll,
        );
        // 交代が成立し場が Cloyster に。
        assert_eq!(result.state.pokemon(Player::P1).name, "Cloyster");
        assert!(result.events.iter().any(|e| matches!(e, Event::Switch { .. })));
        // P2 Cloyster の Shock Wave (でんき) が交代後の Cloyster (みず 2倍) に刺さる。
        assert!(result.state.hp(Player::P1) < result.state.pokemon(Player::P1).max_hp);
    }

    #[test]
    fn faint_requests_forced_switch_then_resolves() {
        // P1 Cloyster を瀕死寸前にして KO させ、強制交代が要求されることを確認。
        let mut state = stage3b_cloyster_mirror();
        state.parties[Player::P1.index()].active_mon_mut().hp = 1;
        // P2 Cloyster の Shock Wave で P1 Cloyster が瀕死。
        let after = apply_turn(
            state,
            Choice::Move(MoveId::ShockWave),
            Choice::Move(MoveId::ShockWave),
            Player::P2,
            &mut MaxRoll,
        );
        assert!(after.state.is_fainted(Player::P1));
        assert!(!after.state.is_done(), "控えが残るので決着していない");
        assert!(after.state.forced_switch[Player::P1.index()]);
        // 強制交代中は P1 の合法手が交代のみ。
        let fc = after.state.legal_choices(Player::P1);
        assert!(fc.iter().all(|c| matches!(c, Choice::Switch(_))));
        assert_eq!(fc.len(), 1);
        // 控え Goodra-Hisui を出す。
        let resolved = apply_forced_switches(
            after.state,
            Some(Choice::Switch(SpeciesId::GoodraHisui.index())),
            None,
        );
        assert_eq!(resolved.state.pokemon(Player::P1).name, "Goodra-Hisui");
        assert!(!resolved.state.forced_switch[Player::P1.index()]);
    }

    #[test]
    fn forced_switch_accessors_track_pending_side() {
        let mut state = stage3b_cloyster_mirror();
        state.parties[Player::P1.index()].active_mon_mut().hp = 1;
        let after = apply_turn(
            state,
            Choice::Move(MoveId::ShockWave),
            Choice::Move(MoveId::ShockWave),
            Player::P2,
            &mut MaxRoll,
        );
        // 共通アクセサが「P1 のみ強制交代待ち」を反映する。
        assert!(after.state.needs_forced_switch(Player::P1));
        assert!(!after.state.needs_forced_switch(Player::P2));
        assert!(after.state.any_forced_switch());
        // 解決後はどちらも待ち状態でない。
        let resolved = apply_forced_switches(
            after.state,
            Some(Choice::Switch(SpeciesId::GoodraHisui.index())),
            None,
        );
        assert!(!resolved.state.needs_forced_switch(Player::P1));
        assert!(!resolved.state.any_forced_switch());
    }

    #[test]
    fn battle_lost_only_when_whole_party_faints() {
        let mut state = stage3b_cloyster_mirror();
        // P1 の両メンバーを瀕死に。
        for i in 0..state.parties[Player::P1.index()].len {
            state.parties[Player::P1.index()].members[i].hp = 0;
        }
        assert!(state.is_lost(Player::P1));
        assert_eq!(state.winner(), Some(Player::P2));
    }

    #[test]
    fn turn_cap_ends_in_draw_with_tie_event() {
        // ターン上限直前から 1 ターン進めると、決着が付かなくても引き分けで終局する。
        // Crunch は高 Def の Cloyster を倒せず瀕死は起きないので勝者は付かない。
        let mut state = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::Cloyster);
        state.turn = MAX_TURNS - 1;
        let result = turn(state, MoveId::Crunch, MoveId::Crunch, Player::P1);
        assert!(result.state.is_done());
        assert!(result.state.is_draw());
        assert_eq!(result.state.winner(), None);
        assert!(result.events.contains(&Event::Tie));
    }

    #[test]
    fn apply_turn_is_noop_after_turn_cap() {
        // 上限到達後の状態に apply_turn を呼んでも何も起きない (is_done ガード)。
        let mut state = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::Cloyster);
        state.turn = MAX_TURNS;
        assert!(state.is_done());
        let result = turn(state, MoveId::Crunch, MoveId::Crunch, Player::P1);
        assert_eq!(result.state.turn, MAX_TURNS);
        assert!(result.events.is_empty());
    }
}
