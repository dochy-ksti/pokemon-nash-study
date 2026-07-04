//! poke-sho-rust と Pokemon Showdown のイベント列パリティ検証 (Stage3a / Stage3b)。
//!
//! 方針 (RNG ストリームは再現しない):
//! 1. ランダムエージェントで Showdown を 1 ゲーム走らせ、射影イベント列 `e_sd` を得る。
//!    `|switch|`/`|turn|` も射影に含める (交代・ターン境界)。
//! 2. `e_sd` をターン単位に区切り、各ターンの両者の `Choice`(技 or 交代)をイベントから
//!    復元して `apply_turn` / `apply_forced_switches` を駆動する。急所は観測値に固定し、
//!    ダメージロール 16 通り (85..=100) を技数ぶん総当たりして観測イベントを再現する。
//! 3. 先発・交代の種族は switch 行から復元する。これにより Goodra と Goodra-Hisui の
//!    forme 識別 (Steel/Dragon vs Dragon) もパリティ検証の対象になる。
//!
//! 同速 (spe=90) なので Showdown 側の行動順は非再現の乱数。行動順は観測 (最初の行動
//! イベントの側) から `first_player` に与える。同一ターンの二重強制交代の順序も乱数なので、
//! 強制交代ブロックは player 順に正規化して比較する。
//!
//! Showdown CLI が無い環境では skip する。

use std::path::PathBuf;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use poke_env_rust::showdown_trait::{ReceiveInfo, Request, SendInfo, create_simulator_game};
use poke_sho_rust::battle::{BattleState, Choice, Player, apply_forced_switches, apply_turn};
use poke_sho_rust::battle_rng::BattleRng;
use poke_sho_rust::event::Event;
use poke_sho_rust::scenario::{MoveId, SpeciesId, Stage, TeamId};

const NUM_GAMES: u32 = 21;

/// Stage3a で巡回する種族組合せ (3 種族の全 9 通り)。Cloyster / Goodra-Hisui /
/// 原種 Goodra。後ろ 2 つは実数値・技構成が同一でタイプのみ異なる forme ミラー。
const STAGE3A_PAIRS: [(SpeciesId, SpeciesId); 9] = [
    (SpeciesId::Cloyster, SpeciesId::Cloyster),
    (SpeciesId::Cloyster, SpeciesId::GoodraHisui),
    (SpeciesId::GoodraHisui, SpeciesId::Cloyster),
    (SpeciesId::GoodraHisui, SpeciesId::GoodraHisui),
    (SpeciesId::Cloyster, SpeciesId::Goodra),
    (SpeciesId::Goodra, SpeciesId::Cloyster),
    (SpeciesId::Goodra, SpeciesId::Goodra),
    (SpeciesId::GoodraHisui, SpeciesId::Goodra),
    (SpeciesId::Goodra, SpeciesId::GoodraHisui),
];

fn showdown_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("pokemon-showdown")
}

/// rust 側の対戦構成。Showdown へ送るチームは別途テキストで渡す。
enum Setup {
    /// 1v1。先発種族は switch 行から復元し、期待値と一致を確認する。
    Stage3a { p1: SpeciesId, p2: SpeciesId },
    /// 交代あり。TeamId は既知、先発 active は switch 行から復元する。
    Stage3b { p1_team: TeamId, p2_team: TeamId },
}

impl Setup {
    fn stage(&self) -> Stage {
        match self {
            Setup::Stage3a { .. } => Stage::Stage3a,
            Setup::Stage3b { .. } => Stage::Stage3b,
        }
    }
}

/// 復元したロール列・急所列を順に返す決定論 RNG。`apply_turn` は技ごとに
/// `is_crit` → `damage_roll` の順で 1 回ずつ引く。
///
/// 観測ロール数 (`n_moves`) を超える引きには飽和値 (roll=100/crit=false) を返す。
/// ロール総当たり中の誤った組合せでは「本来 KO する技が KO せず相手も追撃する」ため
/// 観測より多く技が発火し得るが、そうした試行は期待列と長さが合わず不採用になるだけ。
struct ScriptRng {
    crits: Vec<bool>,
    rolls: Vec<u8>,
    ci: usize,
    ri: usize,
}
impl BattleRng for ScriptRng {
    fn damage_roll(&mut self) -> u8 {
        let r = self.rolls.get(self.ri).copied().unwrap_or(100);
        self.ri += 1;
        r
    }
    fn is_crit(&mut self, _stage: u8) -> bool {
        let c = self.crits.get(self.ci).copied().unwrap_or(false);
        self.ci += 1;
        c
    }
}

/// ランダムエージェントで Showdown を 1 ゲーム駆動し、p1 側の Update イベントを
/// 射影イベント列として収集する。
async fn run_showdown_game(
    seed: [u32; 4],
    agent_seed: u64,
    p1_team_text: String,
    p2_team_text: String,
    team_size: u8,
) -> Vec<Event> {
    let mut driver = create_simulator_game(
        showdown_dir(),
        p1_team_text,
        p2_team_text,
        "gen9customgame",
        seed,
    )
    .await
    .expect("create simulator game");

    let mut e_sd: Vec<Event> = Vec::new();
    let mut rng = Pcg64::seed_from_u64(agent_seed);
    let mut p1_done = false;
    let mut p2_done = false;
    let mut finished = false;

    let timeout = tokio::time::Duration::from_secs(30);
    let run = tokio::time::timeout(timeout, async {
        loop {
            tokio::select! {
                msg = driver.p1.receive(), if !p1_done => {
                    match msg {
                        Ok(ReceiveInfo::Request(req)) => {
                            if let Some(a) = pick_action(&req, &mut rng, team_size) {
                                let _ = driver.p1.send(a).await;
                            }
                        }
                        Ok(ReceiveInfo::Update(update)) => {
                            for ev in update.events {
                                let end = matches!(ev, Event::Win { .. } | Event::Tie);
                                e_sd.push(ev);
                                if end { finished = true; }
                            }
                        }
                        Err(_) => p1_done = true,
                    }
                }
                msg = driver.p2.receive(), if !p2_done => {
                    match msg {
                        Ok(ReceiveInfo::Request(req)) => {
                            if let Some(a) = pick_action(&req, &mut rng, team_size) {
                                let _ = driver.p2.send(a).await;
                            }
                        }
                        Ok(ReceiveInfo::Update(_)) => {}
                        Err(_) => p2_done = true,
                    }
                }
            }
            if finished || (p1_done && p2_done) {
                break;
            }
        }
    })
    .await;
    assert!(run.is_ok(), "showdown game timed out");
    driver.shutdown().await;
    e_sd
}

fn pick_action(req: &Request, rng: &mut Pcg64, team_size: u8) -> Option<SendInfo> {
    match req {
        Request::TeamPreview => {
            // 先発をばらつかせるため team preview 順をシャッフルする。
            let mut order: Vec<u8> = (1..=team_size).collect();
            order.shuffle(rng);
            Some(SendInfo::Team(order))
        }
        Request::Wait => None,
        Request::Move { moves, .. } => {
            let enabled: Vec<usize> = moves
                .iter()
                .enumerate()
                .filter(|(_, m)| !m.disabled)
                .map(|(i, _)| i)
                .collect();
            if enabled.is_empty() {
                return None;
            }
            let slot = enabled[rng.gen_range(0..enabled.len())] + 1;
            Some(SendInfo::Move(slot as u8))
        }
        Request::ForceSwitch { team } => {
            let alive: Vec<usize> = team
                .iter()
                .enumerate()
                .filter(|(_, m)| !m.fainted)
                .map(|(i, _)| i)
                .collect();
            if alive.is_empty() {
                return None;
            }
            let slot = alive[rng.gen_range(0..alive.len())] + 1;
            Some(SendInfo::Switch(slot as u8))
        }
    }
}

fn player_from_num(n: u8) -> Player {
    match n {
        1 => Player::P1,
        _ => Player::P2,
    }
}

fn move_from_id(id: &str) -> MoveId {
    MoveId::from_showdown_id(id)
        .unwrap_or_else(|| panic!("unexpected move id in scenario parity test: {id}"))
}

/// switch 行の species 名 → 種族 ID。forme を厳密に区別する。
fn species_from_name(name: &str) -> SpeciesId {
    SpeciesId::from_species_name(name)
        .unwrap_or_else(|| panic!("unexpected switch species in parity test: {name}"))
}

/// 交代先 species → パーティ絶対 index (build_team_party の slot = SpeciesId.index())。
fn switch_target_index(species: &str) -> usize {
    species_from_name(species).index()
}

/// イベントの行動主体 player (該当しなければ None)。
fn event_player(ev: &Event) -> Option<u8> {
    match ev {
        Event::Move { user, .. } => Some(user.player),
        Event::Switch { who, .. } => Some(who.player),
        _ => None,
    }
}

/// `e_sd` を `Event::Turn` で分割する。返すのは各ターンのイベント範囲 (Turn マーカーは
/// 含めない)。最初の Turn より前 (先発リード) は別途返す。
fn split_turns(e_sd: &[Event]) -> (std::ops::Range<usize>, Vec<std::ops::Range<usize>>) {
    let first_turn = e_sd
        .iter()
        .position(|e| matches!(e, Event::Turn { .. }))
        .expect("e_sd must contain at least one |turn|");
    let lead = 0..first_turn;
    let mut turns = Vec::new();
    let mut i = first_turn;
    while i < e_sd.len() {
        debug_assert!(matches!(e_sd[i], Event::Turn { .. }));
        let start = i + 1;
        let mut j = start;
        while j < e_sd.len() && !matches!(e_sd[j], Event::Turn { .. }) {
            j += 1;
        }
        turns.push(start..j);
        i = j;
    }
    (lead, turns)
}

/// 先発リード領域から各 player の先発 species を取り出す。
fn lead_species(e_sd: &[Event], lead: std::ops::Range<usize>) -> (String, String) {
    let mut p1 = None;
    let mut p2 = None;
    for ev in &e_sd[lead] {
        if let Event::Switch { who, species, .. } = ev {
            match who.player {
                1 => p1 = Some(species.clone()),
                _ => p2 = Some(species.clone()),
            }
        }
    }
    (
        p1.expect("p1 lead switch missing"),
        p2.expect("p2 lead switch missing"),
    )
}

fn build_initial_state(setup: &Setup, e_sd: &[Event], lead: std::ops::Range<usize>) -> BattleState {
    let (p1_sp, p2_sp) = lead_species(e_sd, lead);
    match setup {
        Setup::Stage3a { p1, p2 } => {
            // switch 行から復元した種族が期待値と一致することを確認 (forme 識別の検証)。
            assert_eq!(species_from_name(&p1_sp), *p1, "p1 lead species mismatch");
            assert_eq!(species_from_name(&p2_sp), *p2, "p2 lead species mismatch");
            BattleState::new(Stage::Stage3a, *p1, *p2)
        }
        Setup::Stage3b { p1_team, p2_team } => BattleState::new_with_teams(
            Stage::Stage3b,
            (*p1_team, switch_target_index(&p1_sp)),
            (*p2_team, switch_target_index(&p2_sp)),
        ),
    }
}

/// 強制交代ブロック (最初の Move 以降の Switch 群) を player 昇順に並べた列を返す。
/// 二重強制交代の順序は非再現の乱数なので正規化して比較する。
fn normalize_forced(slice: &[Event], first_move: Option<usize>) -> Vec<Event> {
    let mut out = slice.to_vec();
    if let Some(fm) = first_move {
        let tail = &mut out[fm..];
        // 末尾の連続 Switch 群を player 順に安定ソートする。
        let forced_start = tail
            .iter()
            .position(|e| matches!(e, Event::Switch { .. }))
            .map(|p| fm + p);
        if let Some(fs) = forced_start {
            out[fs..].sort_by_key(|e| event_player(e).unwrap_or(0));
        }
    }
    out
}

fn replay_turn(state: &mut BattleState, slice: &[Event]) -> Result<Vec<Event>, String> {
    if slice.is_empty() {
        return Ok(Vec::new());
    }
    let first_move = slice.iter().position(|e| matches!(e, Event::Move { .. }));

    // 行動順: 最初の行動イベント (Switch/Move) の側を first_player に。これで自発交代・
    // 技いずれの観測順も apply_turn の [first, opp] 反復で再現される。
    let first_player = slice
        .iter()
        .find_map(event_player)
        .map(player_from_num)
        .unwrap_or(Player::P1);

    // 各 player の choice を復元する。
    let choice_for = |p: Player| -> Choice {
        let num = p.index() as u8 + 1;
        // 1. 最初の Move より前の自発交代。
        let voluntary_end = first_move.unwrap_or(slice.len());
        for ev in &slice[..voluntary_end] {
            if let Event::Switch { who, species, .. } = ev {
                if who.player == num {
                    return Choice::Switch(switch_target_index(species));
                }
            }
        }
        // 2. move フェーズの Move。
        for ev in &slice[voluntary_end..] {
            if let Event::Move { user, move_id, .. } = ev {
                if user.player == num {
                    return Choice::Move(move_from_id(move_id));
                }
            }
        }
        // 3. 観測不能 (相手の速攻で瀕死しスキップ)。任意の合法技で埋める。
        state
            .legal_choices(p)
            .into_iter()
            .find(|c| matches!(c, Choice::Move(_)))
            .unwrap_or_else(|| state.legal_choices(p)[0])
    };
    let p1_choice = choice_for(Player::P1);
    let p2_choice = choice_for(Player::P2);

    // 観測した急所列 (Move イベントごと、出現順)。
    let mut crits = Vec::new();
    let mut idx = 0;
    while idx < slice.len() {
        if matches!(slice[idx], Event::Move { .. }) {
            let mut k = idx + 1;
            let mut crit = false;
            while k < slice.len() && !matches!(slice[k], Event::Move { .. }) {
                if matches!(slice[k], Event::Crit { .. }) {
                    crit = true;
                }
                k += 1;
            }
            crits.push(crit);
            idx = k;
        } else {
            idx += 1;
        }
    }
    let n_moves = crits.len();

    // 強制交代の choice (ターン末尾の Switch 群)。
    let forced_for = |p: Player| -> Option<Choice> {
        let num = p.index() as u8 + 1;
        let fm = first_move?;
        slice[fm..].iter().find_map(|ev| match ev {
            Event::Switch { who, species, .. } if who.player == num => {
                Some(Choice::Switch(switch_target_index(species)))
            }
            _ => None,
        })
    };
    let forced_p1 = forced_for(Player::P1);
    let forced_p2 = forced_for(Player::P2);

    let expected = normalize_forced(slice, first_move);

    // ロール総当たり (n_moves 桁・各 85..=100)。
    let mut rolls = vec![85u8; n_moves];
    loop {
        let mut trial = *state;
        let mut rng = ScriptRng {
            crits: crits.clone(),
            rolls: rolls.clone(),
            ci: 0,
            ri: 0,
        };
        let tr = apply_turn(trial, p1_choice, p2_choice, first_player, &mut rng);
        let mut produced = tr.events;
        trial = tr.state;
        let fr = apply_forced_switches(trial, forced_p1, forced_p2);
        produced.extend(fr.events);
        trial = fr.state;

        if produced == expected {
            *state = trial;
            return Ok(produced);
        }

        // オドメータを進める。
        if n_moves == 0 {
            break;
        }
        let mut i = n_moves - 1;
        loop {
            if rolls[i] < 100 {
                rolls[i] += 1;
                break;
            }
            rolls[i] = 85;
            if i == 0 {
                return Err(format!(
                    "no damage roll combo reproduces turn\n  expected: {expected:#?}\n  \
                     p1_choice={p1_choice:?} p2_choice={p2_choice:?} first={first_player:?} \
                     crits={crits:?}"
                ));
            }
            i -= 1;
        }
    }

    Err(format!(
        "turn with no moves did not match\n  expected: {expected:#?}"
    ))
}

fn replay_and_check(e_sd: &[Event], setup: &Setup) -> Result<(), String> {
    let (lead, turns) = split_turns(e_sd);
    let mut state = build_initial_state(setup, e_sd, lead);
    for range in turns {
        replay_turn(&mut state, &e_sd[range])?;
    }
    Ok(())
}

/// g からゲーム構成を選ぶ。3a (forme 識別) と 3b (交代) を両方回す。
fn setup_for_game(g: u32) -> Setup {
    if (g as usize) < STAGE3A_PAIRS.len() {
        // 3a: 3 種族 (Cloyster / Goodra-Hisui / 原種 Goodra) の全 9 組合せを巡回。
        // Goodra-Hisui と原種 Goodra は実数値・技構成が同一でタイプのみ異なるため、
        // switch 行からの forme 識別がパリティ検証の対象になる。
        let (p1, p2) = STAGE3A_PAIRS[g as usize];
        Setup::Stage3a { p1, p2 }
    } else {
        // 3b: (Team1/Team2)^2 を巡回。先発は team preview シャッフルでばらつく。
        let (p1_team, p2_team) = match g % 4 {
            0 => (TeamId::Team1, TeamId::Team1),
            1 => (TeamId::Team1, TeamId::Team2),
            2 => (TeamId::Team2, TeamId::Team1),
            _ => (TeamId::Team2, TeamId::Team2),
        };
        Setup::Stage3b { p1_team, p2_team }
    }
}

fn team_texts(setup: &Setup) -> (String, String, u8) {
    match setup {
        Setup::Stage3a { p1, p2 } => (
            p1.team_text(Stage::Stage3a).to_string(),
            p2.team_text(Stage::Stage3a).to_string(),
            1,
        ),
        Setup::Stage3b { p1_team, p2_team } => (
            p1_team.team_text(Stage::Stage3b).to_string(),
            p2_team.team_text(Stage::Stage3b).to_string(),
            2,
        ),
    }
}

#[tokio::test]
async fn event_streams_match_showdown() {
    if !showdown_dir().join("pokemon-showdown").exists() {
        eprintln!("pokemon-showdown CLI not found, skipping parity test");
        return;
    }

    for g in 0..NUM_GAMES {
        let seed = [g * 4 + 1, g * 4 + 2, g * 4 + 3, g * 4 + 4];
        let agent_seed = 0x9e37_79b9u64.wrapping_add((g as u64).wrapping_mul(2654435761));
        let setup = setup_for_game(g);
        let (p1_text, p2_text, team_size) = team_texts(&setup);
        let e_sd = run_showdown_game(seed, agent_seed, p1_text, p2_text, team_size).await;

        assert!(
            e_sd.iter().any(|e| matches!(e, Event::Move { .. })),
            "game {g}: no moves observed"
        );
        assert!(
            e_sd.iter()
                .any(|e| matches!(e, Event::Win { .. } | Event::Tie)),
            "game {g}: game did not finish"
        );

        if let Err(msg) = replay_and_check(&e_sd, &setup) {
            panic!("game {g} (seed {seed:?}, stage {:?}) parity mismatch:\n{msg}", setup.stage());
        }
    }
}
