//! 現在の学習シナリオの定義: Cloyster (Def 特化) vs Goodra (SpD 特化) の 1v1。
//!
//! 目的は「相手の種族 (パルシェン or ヌメルゴン) を観測して、物理 (かみくだく) と
//! 特殊 (あくのはどう) を切り替える」ことを学習させること。両側それぞれが 2 種族の
//! どちらかをランダムに保持し、対戦組み合わせは 4 通り。
//!
//! ここには登場する種族・技・難易度ステージとそのチームデータ、Showdown 相互変換、
//! および `BattleState::new` (チームテキストから実数ステータスを解決する) を置く。
//! 戦闘の進行ロジックそのものは `battle` モジュールにある。

use crate::battle::{BattleState, MAX_TURNS};
use crate::moves::{
    BULLDOZE, CRUNCH, DARK_PULSE, FAIRY_PHY_60, FIGHT_SPE_60, MoveData, SHOCK_WAVE,
};
use crate::party::{MAX_PARTY, Party, PokemonState};
use crate::team;
use serde::{Deserialize, Serialize};

/// Stage 3a — タイプ相性導入。Cloyster (Water/Ice) と Goodra-Hisui (Steel/Dragon) が
/// 4 技 (Crunch/Dark Pulse/Shock Wave/Bulldoze) を持つ 1v1。相手の弱点を突く技
/// (Cloyster には Shock Wave、Goodra には Bulldoze) を選べるか検証する。
const STAGE3A_CLOYSTER_TEAM: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../team/poke-ai3/scenario/stage3a_cloyster.txt"
));
const STAGE3A_GOODRA_TEAM: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../team/poke-ai3/scenario/stage3a_goodra.txt"
));
/// 原種 Goodra (Dragon 単) の Stage3a チーム。Goodra-Hisui と実数値・技構成を完全に
/// 揃えた (HP157/A117/B94/C117/D222/S90) ミラー。種族ベースが違うため EV/IV/性格で
/// 同じ実数値に合わせている。タイプだけが Hisui と異なり、forme 識別の検証に使う。
const STAGE3A_GOODRA_PLAIN_TEAM: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../team/poke-ai3/scenario/stage3a_goodra_plain.txt"
));
/// Stage 3b — 交代学習。両側とも {Cloyster, Goodra-Hisui} の 2 体パーティ。
/// 技構成は 2 チーム (TeamId) のバリアントで、同種でも持つ技が異なる:
///   Team1: Cloyster=Shock Wave / Goodra-Hisui=Bulldoze
///   Team2: Cloyster=Bulldoze   / Goodra-Hisui=Shock Wave
/// Shock Wave は Cloyster (Water) に、Bulldoze は Goodra-Hisui (Steel) に SE。
/// よって「相手の場の種族に SE が通る自個体を場に出す」交代を学習させる。各サイドは
/// チーム・先発ともに独立ランダム。
const STAGE3B_TEAM1: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../team/poke-ai3/scenario/stage3b_team1.txt"
));
const STAGE3B_TEAM2: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../team/poke-ai3/scenario/stage3b_team2.txt"
));
/// Stage 3c — 対称対面の交代学習。両側とも {Cloyster, 通常 Goodra} の 2 体パーティ。
/// 3b の非対称性 (Shock Wave 半減 / Bulldoze 等倍) を除去し、物理/特殊スプリットと
/// Def↔SpD 反転で完全対称化した。技構成は 2 チーム (TeamId) のバリアント:
///   Team1: Cloyster=FightSpe60 / Goodra=FairyPhy60
///   Team2: Cloyster=FairyPhy60  / Goodra=FightSpe60
/// FightSpe60 (Fighting) は Cloyster (Ice) に、FairyPhy60 (Fairy) は Goodra (Dragon) に SE。
const STAGE3C_TEAM1: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../team/poke-ai3/scenario/stage3c_team1.txt"
));
const STAGE3C_TEAM2: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../team/poke-ai3/scenario/stage3c_team2.txt"
));

/// シナリオの難易度ステージ識別子。タイプ相性 1v1 (Stage3a) と交代 (Stage3b)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Stage {
    /// タイプ相性導入・4 技 1v1。Cloyster (B=222/D=94) vs Goodra-Hisui (B=94/D=222)。
    Stage3a,
    /// 交代学習。両側 {Cloyster, Goodra-Hisui} の 2 体・各 1 技。先発はランダム。
    Stage3b,
    /// 対称対面の交代学習。両側 {Cloyster, 通常 Goodra} の 2 体・各 1 技 (FightSpe60/
    /// FairyPhy60)。3b の非対称性を除去し、ナッシュ均衡の変化を観察する。先発はランダム。
    Stage3c,
}

impl Stage {
    /// Showdown / CLI で使う短い名前 ("3a" / "3b" / "3c")。
    pub fn short_name(self) -> &'static str {
        match self {
            Stage::Stage3a => "3a",
            Stage::Stage3b => "3b",
            Stage::Stage3c => "3c",
        }
    }

    /// 交代を伴う (複数体パーティの) ステージか。
    pub fn is_party(self) -> bool {
        matches!(self, Stage::Stage3b | Stage::Stage3c)
    }

    /// 短い名前からパースする (大文字小文字を無視)。
    pub fn from_short_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "3a" => Some(Stage::Stage3a),
            "3b" => Some(Stage::Stage3b),
            "3c" => Some(Stage::Stage3c),
            _ => None,
        }
    }
}

/// シナリオに登場する 2 種族。観測ベクトルの種族 ID と一致する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum SpeciesId {
    /// 物理耐久側 (Def 高、SpD 低)。Water/Ice。
    Cloyster = 0,
    /// シナリオの特殊耐久側 (Def 低、SpD 高)。Steel/Dragon。原種 Goodra とは
    /// 別種族として厳密に区別する (取り違えがダメージ・タイプ相性のバグを生むため)。
    GoodraHisui = 1,
    /// 原種 Goodra (Dragon 単)。Stage3a では Goodra-Hisui と実数値・技構成を揃えた
    /// ミラーとして登場し、タイプだけが異なる (forme 識別の検証用)。
    Goodra = 2,
}

impl SpeciesId {
    /// 観測 one-hot の幅。原種 Goodra を含む全 variant 数。
    pub const COUNT: usize = 3;

    pub fn index(self) -> usize {
        self as usize
    }

    /// 1v1 ステージの単体チームテキスト。複数体パーティのステージ (Stage3b) では
    /// 単体テキストは存在しないので呼ばない (`is_party` で分岐する)。
    pub fn team_text(self, stage: Stage) -> &'static str {
        match (stage, self) {
            (Stage::Stage3a, SpeciesId::Cloyster) => STAGE3A_CLOYSTER_TEAM,
            (Stage::Stage3a, SpeciesId::GoodraHisui) => STAGE3A_GOODRA_TEAM,
            (Stage::Stage3a, SpeciesId::Goodra) => STAGE3A_GOODRA_PLAIN_TEAM,
            (Stage::Stage3b | Stage::Stage3c, _) => {
                panic!("party scenario (3b/3c); use TeamId::team_text / new_with_teams")
            }
        }
    }

    /// Showdown 種族名 (フォルム含む正式表記。switch DETAILS と一致)。
    pub fn name(self) -> &'static str {
        match self {
            SpeciesId::Cloyster => "Cloyster",
            SpeciesId::GoodraHisui => "Goodra-Hisui",
            SpeciesId::Goodra => "Goodra",
        }
    }

    /// 種族名 (Showdown 表記) から種族 ID へ。フォルムを厳密に区別する:
    /// "Goodra-Hisui"→`GoodraHisui` / "Goodra"→`Goodra` (吸収しない)。
    pub fn from_species_name(name: &str) -> Option<SpeciesId> {
        match name {
            "Cloyster" => Some(SpeciesId::Cloyster),
            "Goodra-Hisui" => Some(SpeciesId::GoodraHisui),
            "Goodra" => Some(SpeciesId::Goodra),
            _ => None,
        }
    }
}

/// Stage3b の技構成バリアント。同種でも持つ技が異なる (`STAGE3B_TEAM1`/`STAGE3B_TEAM2`)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum TeamId {
    Team1,
    Team2,
}

impl TeamId {
    /// このチームのチームテキスト。stage で 3b ({Cloyster, Goodra-Hisui}) と
    /// 3c ({Cloyster, 通常 Goodra}) を切り替える。非パーティステージでは呼ばない。
    pub fn team_text(self, stage: Stage) -> &'static str {
        match (stage, self) {
            (Stage::Stage3b, TeamId::Team1) => STAGE3B_TEAM1,
            (Stage::Stage3b, TeamId::Team2) => STAGE3B_TEAM2,
            (Stage::Stage3c, TeamId::Team1) => STAGE3C_TEAM1,
            (Stage::Stage3c, TeamId::Team2) => STAGE3C_TEAM2,
            (Stage::Stage3a, _) => {
                panic!("Stage3a is not a party scenario; use SpeciesId::team_text")
            }
        }
    }
}

/// 技スロット (全シナリオ共通の固定枠)。Crunch=あく物理, Dark Pulse=あく特殊,
/// Shock Wave=でんき特殊 (対 Cloyster 弱点), Bulldoze=じめん物理 (対 Goodra-Hisui 弱点)。
/// 各個体が実際に覚えている技は `legal_choices` の合法手マスクで絞る。
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
pub enum MoveId {
    Crunch,
    DarkPulse,
    ShockWave,
    Bulldoze,
    /// stage3c: Fighting 特殊 60 (対 Cloyster 弱点)。既存 4 技の後ろ index 4 に追加。
    FightSpe60,
    /// stage3c: Fairy 物理 60 (対 Goodra 弱点)。index 5 に追加。
    FairyPhy60,
}

pub const NUM_MOVES: usize = 6;

impl MoveId {
    pub fn data(self) -> MoveData {
        match self {
            MoveId::Crunch => CRUNCH,
            MoveId::DarkPulse => DARK_PULSE,
            MoveId::ShockWave => SHOCK_WAVE,
            MoveId::Bulldoze => BULLDOZE,
            MoveId::FightSpe60 => FIGHT_SPE_60,
            MoveId::FairyPhy60 => FAIRY_PHY_60,
        }
    }

    pub fn index(self) -> usize {
        match self {
            MoveId::Crunch => 0,
            MoveId::DarkPulse => 1,
            MoveId::ShockWave => 2,
            MoveId::Bulldoze => 3,
            MoveId::FightSpe60 => 4,
            MoveId::FairyPhy60 => 5,
        }
    }

    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(MoveId::Crunch),
            1 => Some(MoveId::DarkPulse),
            2 => Some(MoveId::ShockWave),
            3 => Some(MoveId::Bulldoze),
            4 => Some(MoveId::FightSpe60),
            5 => Some(MoveId::FairyPhy60),
            _ => None,
        }
    }

    /// 全技を index 順に列挙する。
    pub const ALL: [MoveId; NUM_MOVES] = [
        MoveId::Crunch,
        MoveId::DarkPulse,
        MoveId::ShockWave,
        MoveId::Bulldoze,
        MoveId::FightSpe60,
        MoveId::FairyPhy60,
    ];

    /// Showdown の技スロット (1 始まり)。スロット番号は request の技リスト内の
    /// 位置で、リストは `MoveId` の index 順に並ぶ (チームテキストも同順) ため
    /// `slot = index + 1` が常に成立する。
    pub fn showdown_slot(self) -> u8 {
        self.index() as u8 + 1
    }

    /// Showdown の技スロット (1 始まり) から `MoveId` へ。範囲外は `None`。
    pub fn from_showdown_slot(slot: u8) -> Option<Self> {
        Self::from_index((slot as usize).checked_sub(1)?)
    }

    /// Showdown 正規化 id から `MoveId` へ。
    pub fn from_showdown_id(id: &str) -> Option<Self> {
        match id {
            "crunch" => Some(MoveId::Crunch),
            "darkpulse" => Some(MoveId::DarkPulse),
            "shockwave" => Some(MoveId::ShockWave),
            "bulldoze" => Some(MoveId::Bulldoze),
            "fightspe60" => Some(MoveId::FightSpe60),
            "fairyphy60" => Some(MoveId::FairyPhy60),
            _ => None,
        }
    }
}

impl BattleState {
    /// 1v1 ステージ (Stage3a) のバトル状態を作る。各サイド 1 体・交代なし。
    /// パーティステージ (Stage3b) は技構成 (TeamId) が要るので `new_with_teams` を使う。
    pub fn new(stage: Stage, p1: SpeciesId, p2: SpeciesId) -> Self {
        assert!(
            !stage.is_party(),
            "party scenario (Stage3b) must use BattleState::new_with_teams"
        );
        BattleState {
            parties: [build_single_party(stage, p1), build_single_party(stage, p2)],
            turn: 0,
            forced_switch: [false, false],
            max_turns: MAX_TURNS,
        }
    }

    /// パーティステージ (Stage3b) のバトル状態を作る。各サイドに技構成 `TeamId` と
    /// 先発 (場に出す `active` index) を指定する。両サイドはチーム・先発とも独立。
    pub fn new_with_teams(
        stage: Stage,
        p1: (TeamId, usize),
        p2: (TeamId, usize),
    ) -> Self {
        assert!(
            stage.is_party(),
            "non-party scenario must use BattleState::new"
        );
        BattleState {
            parties: [
                build_team_party(stage, p1.0, p1.1),
                build_team_party(stage, p2.0, p2.1),
            ],
            turn: 0,
            forced_switch: [false, false],
            max_turns: MAX_TURNS,
        }
    }
}

/// 1v1 ステージの単体パーティを構築する (`lead` 単体・交代不可)。
fn build_single_party(stage: Stage, lead: SpeciesId) -> Party {
    let resolved =
        team::resolve_first(lead.team_text(stage)).expect("scenario team text should parse");
    Party::single(PokemonState::from_resolved(lead, &resolved))
}

/// パーティステージ (3b/3c) の 2 体パーティを `team` の技構成で構築し、`active` を場に
/// 出す。メンバーは **team ファイルの宣言順**に配置する (`active`/lead も宣言順 index)。
/// 旧実装は `slots[sid.index()]` で species index 順に置いていたが、通常 Goodra は
/// SpeciesId=2 で MAX_PARTY=2 を超え overflow するため宣言順配置に変更した。3b は
/// team ファイル先頭が Cloyster (index 0)・次が Goodra-Hisui (index 1) で宣言順と
/// species 順が一致するため挙動不変。
fn build_team_party(stage: Stage, team: TeamId, active: usize) -> Party {
    let sets = team::resolve_team(team.team_text(stage)).expect("party team text should parse");
    let mut first: Option<PokemonState> = None;
    let mut slots: [Option<PokemonState>; MAX_PARTY] = [None; MAX_PARTY];
    let mut len = 0;
    for set in &sets {
        let sid = SpeciesId::from_species_name(set.species.name)
            .expect("party species must map to a SpeciesId");
        let mon = PokemonState::from_resolved(sid, set);
        assert!(len < MAX_PARTY, "party team exceeds MAX_PARTY");
        slots[len] = Some(mon);
        first.get_or_insert(mon);
        len += 1;
    }
    let fill = first.expect("party team must have at least one member");
    let mut arr = [fill; MAX_PARTY];
    for (i, slot) in slots.iter().enumerate() {
        if let Some(mon) = slot {
            arr[i] = *mon;
        }
    }
    assert!(active < len, "party active index out of range");
    Party {
        members: arr,
        len,
        active,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::battle::Player;

    #[test]
    fn stage3a_cloyster_and_goodra_hisui_are_mirrors() {
        let st = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::GoodraHisui);
        let cloy = st.pokemon(Player::P1);
        let goodra = st.pokemon(Player::P2);
        // Cloyster: HP157/A117/B222/C117/D94/S90.
        assert_eq!((cloy.stats.def, cloy.stats.spd), (222, 94));
        // Goodra-Hisui: 鏡像で B94/D222、種族名と type も Hisui (Steel/Dragon)。
        assert_eq!((goodra.stats.def, goodra.stats.spd), (94, 222));
        assert_eq!(goodra.name, "Goodra-Hisui");
        assert_eq!(cloy.stats.spe, goodra.stats.spe);
        // 両者 4 技を覚えている。
        assert_eq!(cloy.moves.iter().filter(|m| **m).count(), 4);
        assert_eq!(goodra.moves.iter().filter(|m| **m).count(), 4);
    }

    #[test]
    fn stage3a_plain_goodra_mirrors_hisui_stats_but_differs_in_type() {
        let st = BattleState::new(Stage::Stage3a, SpeciesId::GoodraHisui, SpeciesId::Goodra);
        let hisui = st.pokemon(Player::P1);
        let plain = st.pokemon(Player::P2);
        // 実数値は完全一致 (HP157/A117/B94/C117/D222/S90)。
        assert_eq!(hisui.stats, plain.stats);
        // 技構成も一致 (4 技)。
        assert_eq!(hisui.moves, plain.moves);
        // 種族名・タイプだけが異なる (forme 識別の根拠)。
        assert_eq!(hisui.name, "Goodra-Hisui");
        assert_eq!(plain.name, "Goodra");
        assert_ne!(hisui.types, plain.types);
    }

    #[test]
    fn stage3b_two_teams_swap_moves_per_species() {
        // Team1: Cloyster=Shock Wave / Goodra-Hisui=Bulldoze。
        let t1 = BattleState::new_with_teams(
            Stage::Stage3b,
            (TeamId::Team1, 0),
            (TeamId::Team1, 1),
        );
        let t1_cloy = t1.pokemon(Player::P1);
        let t1_goodra = t1.pokemon(Player::P2);
        assert_eq!(t1_cloy.name, "Cloyster");
        assert!(t1_cloy.moves[MoveId::ShockWave.index()]);
        assert!(!t1_cloy.moves[MoveId::Bulldoze.index()]);
        assert!(t1_goodra.moves[MoveId::Bulldoze.index()]);
        assert!(!t1_goodra.moves[MoveId::ShockWave.index()]);
        // Team2: 技を入れ替え (Cloyster=Bulldoze / Goodra-Hisui=Shock Wave)。
        let t2 = BattleState::new_with_teams(
            Stage::Stage3b,
            (TeamId::Team2, 0),
            (TeamId::Team2, 1),
        );
        let t2_cloy = t2.pokemon(Player::P1);
        let t2_goodra = t2.pokemon(Player::P2);
        assert!(t2_cloy.moves[MoveId::Bulldoze.index()]);
        assert!(!t2_cloy.moves[MoveId::ShockWave.index()]);
        assert!(t2_goodra.moves[MoveId::ShockWave.index()]);
        assert!(!t2_goodra.moves[MoveId::Bulldoze.index()]);
        // 各個体は技 1 つだけ。ステータスはチーム間で不変 (技だけ違う)。
        assert_eq!(t1_cloy.moves.iter().filter(|m| **m).count(), 1);
        assert_eq!((t1_cloy.stats.def, t1_cloy.stats.spd), (t2_cloy.stats.def, t2_cloy.stats.spd));
    }

    #[test]
    fn stage3b_active_index_selects_lead() {
        // active=1 で Goodra-Hisui が先発。
        let st = BattleState::new_with_teams(
            Stage::Stage3b,
            (TeamId::Team1, 1),
            (TeamId::Team2, 0),
        );
        assert_eq!(st.pokemon(Player::P1).name, "Goodra-Hisui");
        assert_eq!(st.pokemon(Player::P2).name, "Cloyster");
    }

    #[test]
    fn stage3c_two_teams_swap_symmetric_moves_per_species() {
        // Team1: Cloyster=FightSpe60 / Goodra=FairyPhy60。
        let t1 = BattleState::new_with_teams(Stage::Stage3c, (TeamId::Team1, 0), (TeamId::Team1, 1));
        let cloy = t1.pokemon(Player::P1);
        let goodra = t1.pokemon(Player::P2);
        assert_eq!(cloy.name, "Cloyster");
        assert_eq!(goodra.name, "Goodra"); // 通常 Goodra (純 Dragon)。Hisui ではない。
        assert!(cloy.moves[MoveId::FightSpe60.index()]);
        assert!(goodra.moves[MoveId::FairyPhy60.index()]);
        // Team2 で技を入れ替え。
        let t2 = BattleState::new_with_teams(Stage::Stage3c, (TeamId::Team2, 0), (TeamId::Team2, 1));
        assert!(t2.pokemon(Player::P1).moves[MoveId::FairyPhy60.index()]);
        assert!(t2.pokemon(Player::P2).moves[MoveId::FightSpe60.index()]);
        // 各個体は技 1 つだけ。
        assert_eq!(cloy.moves.iter().filter(|m| **m).count(), 1);
        assert_eq!(goodra.moves.iter().filter(|m| **m).count(), 1);
    }

    #[test]
    fn stage3c_def_spd_are_mirror_reversed() {
        let st = BattleState::new_with_teams(Stage::Stage3c, (TeamId::Team1, 0), (TeamId::Team1, 1));
        let cloy = st.pokemon(Player::P1);
        let goodra = st.pokemon(Player::P2);
        // Cloyster: B222/D94, Goodra: B94/D222 の鏡像。HP/Atk/SpA/Spe は同値。
        assert_eq!((cloy.stats.def, cloy.stats.spd), (222, 94));
        assert_eq!((goodra.stats.def, goodra.stats.spd), (94, 222));
        assert_eq!(cloy.stats.hp, goodra.stats.hp);
        assert_eq!(cloy.stats.atk, goodra.stats.atk);
        assert_eq!(cloy.stats.spa, goodra.stats.spa);
        assert_eq!(cloy.stats.spe, goodra.stats.spe);
    }

    #[test]
    fn stage3c_declaration_order_places_goodra_at_index_1() {
        // 通常 Goodra は SpeciesId=2 だが、宣言順 (Cloyster→Goodra) で lead=1 が Goodra。
        let st = BattleState::new_with_teams(Stage::Stage3c, (TeamId::Team1, 1), (TeamId::Team2, 0));
        assert_eq!(st.pokemon(Player::P1).name, "Goodra");
        assert_eq!(st.pokemon(Player::P2).name, "Cloyster");
    }

    #[test]
    fn stage_short_name_round_trips() {
        assert_eq!(Stage::Stage3a.short_name(), "3a");
        assert_eq!(Stage::Stage3b.short_name(), "3b");
        assert_eq!(Stage::from_short_name("3A"), Some(Stage::Stage3a));
        assert_eq!(Stage::from_short_name("3b"), Some(Stage::Stage3b));
        assert_eq!(Stage::Stage3c.short_name(), "3c");
        assert_eq!(Stage::from_short_name("3C"), Some(Stage::Stage3c));
        assert_eq!(Stage::from_short_name("2a"), None);
    }

    #[test]
    fn showdown_slot_round_trips_with_index() {
        for idx in 0..NUM_MOVES {
            let mv = MoveId::from_index(idx).unwrap();
            assert_eq!(mv.showdown_slot(), idx as u8 + 1);
            assert_eq!(MoveId::from_showdown_slot(mv.showdown_slot()), Some(mv));
        }
        assert_eq!(MoveId::from_showdown_slot(0), None);
        assert_eq!(MoveId::from_showdown_slot(NUM_MOVES as u8 + 1), None);
    }
}
