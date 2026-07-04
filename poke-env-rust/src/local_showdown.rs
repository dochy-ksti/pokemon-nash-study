//! `poke-sho-rust` を `ShowdownTrait` 越しに使う in-process バックエンド。
//!
//! - `SimulatorShowdown` (subprocess) と同じインタフェースを提供する。
//! - 内部 engine タスクが `BattleState` を保持し、両 player の `SendInfo::Move`
//!   を集めてから `apply_turn` を呼び、結果イベントを Showdown プロトコル形式の
//!   `Update` / `Request` として両 side に流す。
//! - 本シナリオでは両者 Spe=90 で speed tie のため、行動順は seed 由来の擬似乱数で決定する。

use crate::showdown_trait::{
    BattleView, MoveOption, ReceiveInfo, Request, SendInfo, ShowdownError, ShowdownTrait,
    TeamMember, Update,
};
use crate::battle_chacha::BattleChaCha;
use poke_sho_rust::event::PokemonRef;
use poke_sho_rust::battle::{BattleState, Choice, Player, apply_forced_switches, apply_turn};
use poke_sho_rust::scenario::{MoveId, NUM_MOVES};
#[cfg(test)]
use poke_sho_rust::scenario::{SpeciesId, Stage, TeamId};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub struct LocalShowdown {
    side: Player,
    rx: mpsc::UnboundedReceiver<ReceiveInfo>,
    tx: mpsc::UnboundedSender<(Player, SendInfo)>,
    /// engine が手番開始ごとに更新する真の `BattleState`。lookahead はこれを起点にする。
    state: Arc<Mutex<BattleState>>,
}

impl ShowdownTrait for LocalShowdown {
    async fn receive(&mut self) -> Result<ReceiveInfo, ShowdownError> {
        self.rx.recv().await.ok_or(ShowdownError::Closed)
    }

    async fn send(&mut self, send_info: SendInfo) -> Result<(), ShowdownError> {
        self.tx
            .send((self.side, send_info))
            .map_err(|_| ShowdownError::Closed)
    }

    fn battle_state(&self) -> Option<BattleState> {
        Some(*self.state.lock().expect("battle state mutex poisoned"))
    }
}

/// in-process バックエンドの 1 ゲーム分の駆動オブジェクト。
pub struct LocalDriver {
    pub p1: BattleView<LocalShowdown>,
    pub p2: BattleView<LocalShowdown>,
    _engine: EngineGuard,
}

struct EngineGuard {
    handle: Option<JoinHandle<()>>,
}

impl Drop for EngineGuard {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            h.abort();
        }
    }
}

/// ローカル試合を作成する。`seed` は行動順タイブレイクとダメージ乱数用。
/// `randomize` で 16 段ダメージ乱数、`crit_enabled` で急所を切り替える。
/// `initial` は呼び出し側が構築した初期バトル状態 (stage/チーム/先発を内包する)。
pub fn create_local_game(
    seed: [u32; 4],
    randomize: bool,
    crit_enabled: bool,
    initial: BattleState,
) -> LocalDriver {
    let (p1_recv_tx, p1_recv_rx) = mpsc::unbounded_channel();
    let (p2_recv_tx, p2_recv_rx) = mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();

    let state = Arc::new(Mutex::new(initial));

    let p1_show = LocalShowdown {
        side: Player::P1,
        rx: p1_recv_rx,
        tx: cmd_tx.clone(),
        state: state.clone(),
    };
    let p2_show = LocalShowdown {
        side: Player::P2,
        rx: p2_recv_rx,
        tx: cmd_tx,
        state: state.clone(),
    };

    let handle = tokio::spawn(run_engine(
        p1_recv_tx,
        p2_recv_tx,
        cmd_rx,
        seed,
        randomize,
        crit_enabled,
        initial,
        state,
    ));

    LocalDriver {
        p1: BattleView::new(p1_show),
        p2: BattleView::new(p2_show),
        _engine: EngineGuard {
            handle: Some(handle),
        },
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_engine(
    p1_tx: mpsc::UnboundedSender<ReceiveInfo>,
    p2_tx: mpsc::UnboundedSender<ReceiveInfo>,
    mut cmd_rx: mpsc::UnboundedReceiver<(Player, SendInfo)>,
    seed: [u32; 4],
    randomize: bool,
    crit_enabled: bool,
    initial: BattleState,
    shared_state: Arc<Mutex<BattleState>>,
) {
    let mut state = initial;
    let publish = |s: &BattleState| {
        *shared_state.lock().expect("battle state mutex poisoned") = *s;
    };
    publish(&state);
    // ダメージ乱数・急所・速度タイをすべて 1 本の ChaCha8 ストリームから引く。
    // seed から完全再現可能で、毎ターンの速度タイも公平にばらける。
    let mut rng = BattleChaCha::from_seed_words(seed, randomize, crit_enabled);

    if send_request(&p1_tx, &state, Player::P1).is_err() {
        return;
    }
    if send_request(&p2_tx, &state, Player::P2).is_err() {
        return;
    }

    let mut p1_choice: Option<Choice> = None;
    let mut p2_choice: Option<Choice> = None;
    // 強制交代サブフェーズか (true の間は forced 側の交代手だけを待つ)。
    let mut forced_phase = false;

    while !state.is_done() {
        let Some((player, send)) = cmd_rx.recv().await else {
            return;
        };
        let choice = match send {
            SendInfo::Move(slot) => Some(Choice::Move(slot_to_move(slot))),
            // `switch N`(1 始まりパーティ位置)→ パーティ index。
            SendInfo::Switch(n) => Some(Choice::Switch((n.max(1) - 1) as usize)),
            SendInfo::Team(_) | SendInfo::Default => None,
        };
        match player {
            Player::P1 => p1_choice = choice,
            Player::P2 => p2_choice = choice,
        }

        if forced_phase {
            // forced 側が選び終わったら強制交代を解決して通常フェーズへ戻る。
            let ready = (!state.needs_forced_switch(Player::P1) || p1_choice.is_some())
                && (!state.needs_forced_switch(Player::P2) || p2_choice.is_some());
            if !ready {
                continue;
            }
            let result = apply_forced_switches(state, p1_choice, p2_choice);
            state = result.state;
            publish(&state);
            broadcast_update(&p1_tx, &p2_tx, result.events);
            p1_choice = None;
            p2_choice = None;
            forced_phase = false;
            if !state.is_done() && request_both(&p1_tx, &p2_tx, &state).is_err() {
                return;
            }
            continue;
        }

        if let (Some(c1), Some(c2)) = (p1_choice, p2_choice) {
            // 両者 Spe=90 同速 — 毎ターン公平なコインで先手を決める。
            let first = rng.first_player();
            let result = apply_turn(state, c1, c2, first, &mut rng);
            state = result.state;
            publish(&state);
            broadcast_update(&p1_tx, &p2_tx, result.events);
            p1_choice = None;
            p2_choice = None;
            if state.is_done() {
                continue;
            }
            // 瀕死で控えが残る側へは強制交代を要求し、forced フェーズへ移行する。
            if state.any_forced_switch() {
                forced_phase = true;
                if send_force_switch_requests(&p1_tx, &p2_tx, &state).is_err() {
                    return;
                }
            } else if request_both(&p1_tx, &p2_tx, &state).is_err() {
                return;
            }
        }
    }
}

fn slot_to_move(slot: u8) -> MoveId {
    MoveId::from_showdown_slot(slot).unwrap_or(MoveId::Crunch)
}

fn broadcast_update(
    p1_tx: &mpsc::UnboundedSender<ReceiveInfo>,
    p2_tx: &mpsc::UnboundedSender<ReceiveInfo>,
    events: Vec<poke_sho_rust::event::Event>,
) {
    let update = ReceiveInfo::Update(Update { events });
    // 送信失敗 = 受信側 drop (切断)。ここで無視しても直後の request 送信が
    // 失敗してループが終了するため、生存側への送信を優先して握りつぶす。
    let _ = p1_tx.send(update.clone());
    let _ = p2_tx.send(update);
}

fn request_both(
    p1_tx: &mpsc::UnboundedSender<ReceiveInfo>,
    p2_tx: &mpsc::UnboundedSender<ReceiveInfo>,
    state: &BattleState,
) -> Result<(), ()> {
    send_request(p1_tx, state, Player::P1)?;
    send_request(p2_tx, state, Player::P2)
}

/// forced 側にだけ `ForceSwitch` を送る。待たない側には何も送らない (Wait 相当)。
fn send_force_switch_requests(
    p1_tx: &mpsc::UnboundedSender<ReceiveInfo>,
    p2_tx: &mpsc::UnboundedSender<ReceiveInfo>,
    state: &BattleState,
) -> Result<(), ()> {
    for (tx, p) in [(p1_tx, Player::P1), (p2_tx, Player::P2)] {
        if state.needs_forced_switch(p) {
            let team = team_members(state, p);
            tx.send(ReceiveInfo::Request(Request::ForceSwitch { team }))
                .map_err(|_| ())?;
        }
    }
    Ok(())
}

fn send_request(
    tx: &mpsc::UnboundedSender<ReceiveInfo>,
    state: &BattleState,
    player: Player,
) -> Result<(), ()> {
    // 本シナリオはアクティブ 1 体・両技とも常に選択可能。技リストは MoveId の
    // index 順 (= slot 順) で並べる。
    let moves = (0..NUM_MOVES)
        .filter_map(MoveId::from_index)
        .map(|m| MoveOption {
            id: m.data().id.to_string(),
            disabled: false,
        })
        .collect();
    let team = team_members(state, player);
    tx.send(ReceiveInfo::Request(Request::Move { moves, team }))
        .map_err(|_| ())
}

/// `player` のパーティ全員を `TeamMember` 列にする (交代先選択に必要)。
/// 並びはパーティ index 順 = Showdown `switch N` の N。
fn team_members(state: &BattleState, player: Player) -> Vec<TeamMember> {
    let party = state.party(player);
    (0..party.len)
        .map(|i| {
            let mon = &party.members[i];
            let fainted = mon.hp <= 0;
            TeamMember {
                mon: PokemonRef::new(player.index() as u8 + 1, mon.name),
                hp: if fainted { 0 } else { mon.hp.max(0) as u32 },
                max_hp: if fainted { 0 } else { mon.max_hp as u32 },
                fainted,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn run_local_battle_to_end() {
        // Stage 2a、Cloyster vs Goodra。両者 Dark Pulse (slot 2) を撃ち続ければ
        // 数ターンで決着するはず (Cloyster の SpD=81 にダメージが大きい)。
        let mut driver = create_local_game(
            [1, 2, 3, 4],
            true,
            true,
            BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::GoodraHisui),
        );
        let mut winner: Option<String> = None;

        let timeout = tokio::time::Duration::from_secs(5);
        let result = tokio::time::timeout(timeout, async {
            loop {
                tokio::select! {
                    msg = driver.p1.receive() => {
                        if !handle(msg, &mut driver.p1, &mut winner).await { break; }
                    }
                    msg = driver.p2.receive() => {
                        if !handle(msg, &mut driver.p2, &mut winner).await { break; }
                    }
                }
                if winner.is_some() {
                    break;
                }
            }
        })
        .await;
        assert!(result.is_ok(), "local battle timed out");
        assert!(winner.is_some(), "no winner produced");
    }

    async fn handle(
        msg: Result<ReceiveInfo, ShowdownError>,
        view: &mut BattleView<LocalShowdown>,
        winner: &mut Option<String>,
    ) -> bool {
        let Ok(msg) = msg else { return false };
        match msg {
            ReceiveInfo::Request(req) => {
                match req {
                    // 瀕死後は最初の生存控えへ交代する。
                    Request::ForceSwitch { team } => {
                        let slot = team.iter().position(|m| !m.fainted).unwrap_or(0) + 1;
                        let _ = view.send(SendInfo::Switch(slot as u8)).await;
                    }
                    // 通常ターンは Dark Pulse (slot 2)。
                    _ => {
                        let _ = view.send(SendInfo::Move(2)).await;
                    }
                }
                true
            }
            ReceiveInfo::Update(update) => {
                for ev in &update.events {
                    if let poke_sho_rust::event::Event::Win { player } = ev {
                        *winner = Some(player.clone());
                    }
                }
                true
            }
        }
    }

    #[tokio::test]
    async fn stage3b_battle_resolves_through_forced_switch() {
        // Stage3b、両者 2 体パーティ。KO ごとに強制交代を経て最後に決着する。
        // forced-switch サブフェーズが正しく回ることを統合的に確認する。
        let mut driver = create_local_game(
            [9, 7, 5, 3],
            true,
            true,
            BattleState::new_with_teams(Stage::Stage3b, (TeamId::Team1, 0), (TeamId::Team2, 0)),
        );
        let mut winner: Option<String> = None;
        let timeout = tokio::time::Duration::from_secs(5);
        let result = tokio::time::timeout(timeout, async {
            loop {
                tokio::select! {
                    msg = driver.p1.receive() => {
                        if !handle(msg, &mut driver.p1, &mut winner).await { break; }
                    }
                    msg = driver.p2.receive() => {
                        if !handle(msg, &mut driver.p2, &mut winner).await { break; }
                    }
                }
                if winner.is_some() {
                    break;
                }
            }
        })
        .await;
        assert!(result.is_ok(), "stage3b battle timed out (forced-switch deadlock?)");
        assert!(winner.is_some(), "no winner produced in stage3b battle");
    }
}
