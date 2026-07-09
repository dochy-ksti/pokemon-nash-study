//! Pokemon Showdown 互換シミュレータを `ShowdownTrait` 経由で駆動するゲームタスク。
//!
//! 設計 (lookahead 学習版):
//! - 1 ゲーム = 1 シミュレータ + 2 player runner + 1 推論ルータ。
//! - 各 player runner は自分の手番ごとに、真の `BattleState` を起点に lookahead
//!   (`run_lookahead`) を実行し、policy 教師 (`training_pi`) と value 教師を学習
//!   サンプルとして記録する。実際の着手は `selection_pi` からサンプルして進める。
//! - lookahead 内の各 ply の policy/value 推論は `InferenceClient` 経由で root に
//!   batch 送信され、応答は推論ルータが `request_id` で対応する rollout に届ける。
//! - lookahead は真の状態を要する Local バックエンド限定。Showdown バックエンドでは
//!   event 由来の観測で 1 回だけ推論して着手する (互換検証用、学習サンプルは縮退)。

use crate::async_executor::{Backend, InferencedDataItem, ACTION_DIM};
use crate::inference_client::{InferenceClient, PendingMap};
use crate::root_task::{RootEnumFromGame, Trajectory, TrajectoryItem};
use crate::rule_agent::rule_choice;
use poke_env_rust::local_showdown::create_local_game;
use poke_env_rust::observation::{
    BattleState, Choice, Event, MoveId, NUM_BENCH, Player, SpeciesId, Stage, StateForPlayer,
    TeamId, action_index, move_gid_of, species_meta,
};
use poke_env_rust::lookahead::{LookaheadConfig, run_lookahead};
use poke_env_rust::observation::observation_for;
use poke_env_rust::oracle::PolicyOracle;
use poke_env_rust::showdown_trait::{
    BattleView, ReceiveInfo, Request, SendInfo, ShowdownTrait, create_simulator_game,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

fn showdown_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("pokemon-showdown")
}

pub struct GameReceiver {
    receiver: UnboundedReceiver<InferencedDataItem>,
}

impl GameReceiver {
    pub fn new(receiver: UnboundedReceiver<InferencedDataItem>) -> Self {
        Self { receiver }
    }
}

/// 1 試合の初期構成。`p1_species`/`p2_species` は先発 (場の) 種族で、Showdown
/// 経路の観測組み立てに使う。`initial` は真の初期バトル状態、`p1_team_text`/
/// `p2_team_text` は Showdown バックエンド用のチームテキスト。
struct GameConfig {
    p1_species: SpeciesId,
    p2_species: SpeciesId,
    initial: BattleState,
    p1_team_text: String,
    p2_team_text: String,
}

/// SpeciesId index (Cloyster=0, GoodraHisui=1) から SpeciesId へ。
/// パーティ宣言順インデックス (0=Cloyster, 1=2 体目) から先発種族を引く。3b は 2 体目が
/// Goodra-Hisui、3c は通常 Goodra。3a も含め idx 0 は常に Cloyster。
fn species_from_index(stage: Stage, idx: usize) -> SpeciesId {
    match idx {
        0 => SpeciesId::Cloyster,
        _ if stage == Stage::Stage3c => SpeciesId::Goodra,
        _ => SpeciesId::GoodraHisui,
    }
}

/// game_id と iter から初期構成を決定論的に選ぶ。
/// 1v1 ステージは両側の先発種族 4 通りを巡回。Stage3b は各側 (チーム×先発)=4 通り、
/// 両側で 16 通りを巡回し、`new_with_teams` で初期状態を組む。
fn pick_config(game_id: usize, iter: u32, stage: Stage) -> GameConfig {
    let mix = (game_id as u32).wrapping_add(iter.wrapping_mul(0x9e37_79b9));
    if !stage.is_party() {
        let k = mix % 4;
        let p1 = species_from_index(stage, (k as usize) / 2);
        let p2 = species_from_index(stage, (k as usize) % 2);
        return GameConfig {
            p1_species: p1,
            p2_species: p2,
            initial: BattleState::new(stage, p1, p2),
            p1_team_text: p1.team_text(stage).to_string(),
            p2_team_text: p2.team_text(stage).to_string(),
        };
    }
    // Stage3b: 両者は必ず Team1 vs Team2 (クロスチーム) で対戦する。チーム設計上
    // クロスチームなら常にどちらか一方だけが SE 可 (= SE可は厳密に有利) になる。
    // 同チーム対戦は両者SE/双方非SEの縮退局面を生むため作らない。
    // 8 通り = swap(どちらのプレイヤーが Team1 か, bit2) × active1(bit0) × active2(bit1)。
    let k = (mix % 8) as usize;
    let p1_is_team1 = (k & 0b100) == 0;
    let team1 = if p1_is_team1 { TeamId::Team1 } else { TeamId::Team2 };
    let team2 = if p1_is_team1 { TeamId::Team2 } else { TeamId::Team1 };
    let active1 = k & 0b01;
    let active2 = (k >> 1) & 0b01;
    GameConfig {
        p1_species: species_from_index(stage, active1),
        p2_species: species_from_index(stage, active2),
        initial: BattleState::new_with_teams(stage, (team1, active1), (team2, active2)),
        p1_team_text: team1.team_text(stage).to_string(),
        p2_team_text: team2.team_text(stage).to_string(),
    }
}

/// run 全体のシードと (game_id, iter) から対戦シードを導出する。
/// `(game_id << 32) | iter` を奇数定数倍 (全単射) して SplitMix64 で撹拌するため、
/// 同一 run 内でゲーム間・イテレーション間のシード衝突は起きない。
fn battle_seed(run_seed: u64, game_id: usize, iter: u32) -> [u32; 4] {
    let key = (((game_id as u64) << 32) | iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut state = run_seed ^ key;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let (a, b) = (next(), next());
    [(a >> 32) as u32, a as u32, (b >> 32) as u32, b as u32]
}

#[allow(clippy::too_many_arguments)]
pub async fn game_start(
    game_id: usize,
    run_seed: u64,
    backend: Backend,
    randomize: bool,
    crit_enabled: bool,
    stage: Stage,
    lookahead: LookaheadConfig,
    eval_rule_opponent: bool,
    eval_rule_p1: bool,
    roles: Arc<Vec<AtomicU8>>,
    sender: UnboundedSender<RootEnumFromGame>,
    receiver: GameReceiver,
) {
    let mut inference_rx = receiver.receiver;
    let mut iter: u32 = 0;
    loop {
        iter = iter.wrapping_add(1);
        // 敵混合学習用の役割をバトル開始時に読む
        // (0=自己対戦, 1=敵 policy-only, 2=敵 先読みあり)。
        // ブロック境界で Python が書き換えるが、in-flight バトルは開始時の役割を保つ。
        let role = roles
            .get(game_id)
            .map(|r| r.load(Ordering::Relaxed))
            .unwrap_or(0);
        let enemy_game = role != 0;
        let enemy_lookahead = role == 2;
        let seed = battle_seed(run_seed, game_id, iter);
        let cfg = pick_config(game_id, iter, stage);
        let GameConfig {
            p1_species,
            p2_species,
            initial,
            p1_team_text,
            p2_team_text,
        } = cfg;
        let cont = match backend {
            Backend::Local => {
                let d = create_local_game(seed, randomize, crit_enabled, initial);
                run_one_game(
                    game_id, iter, p1_species, p2_species, stage, lookahead, eval_rule_opponent,
                    eval_rule_p1, enemy_game, enemy_lookahead, d.p1, d.p2, &sender,
                    &mut inference_rx,
                )
                .await
            }
            Backend::Showdown => {
                let d = match create_simulator_game(
                    showdown_dir(),
                    p1_team_text,
                    p2_team_text,
                    "gen9customgame",
                    seed,
                )
                .await
                {
                    Ok(d) => d,
                    Err(_) => return,
                };
                run_one_game(
                    game_id, iter, p1_species, p2_species, stage, lookahead, eval_rule_opponent,
                    eval_rule_p1, enemy_game, enemy_lookahead, d.p1, d.p2, &sender,
                    &mut inference_rx,
                )
                .await
            }
        };
        if !cont {
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_one_game<T: ShowdownTrait + Send + 'static>(
    game_id: usize,
    iter: u32,
    p1_species: SpeciesId,
    p2_species: SpeciesId,
    stage: Stage,
    lookahead: LookaheadConfig,
    eval_rule_opponent: bool,
    eval_rule_p1: bool,
    enemy_game: bool,
    enemy_lookahead: bool,
    p1_view: BattleView<T>,
    p2_view: BattleView<T>,
    sender: &UnboundedSender<RootEnumFromGame>,
    inference_rx: &mut UnboundedReceiver<InferencedDataItem>,
) -> bool {
    let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
    let client = InferenceClient::new(game_id, iter, sender.clone(), pending.clone());

    let p1_rng =
        ChaCha8Rng::seed_from_u64(((game_id as u64) << 32) ^ (iter as u64) ^ 0x5051_5031);
    let p2_rng =
        ChaCha8Rng::seed_from_u64(((game_id as u64) << 32) ^ (iter as u64) ^ 0x5051_5032);
    // 評価モードでは P1 = 学習 AI (lookahead)、P2 = 固定ルール方策。
    // 敵ゲームでは既定で P2 (凍結した過去 checkpoint) を policy-only にして高速化する
    // (学習者 P1 は常に探索)。gate/最終評価も policy-only なのでメトリクス整合。
    // enemy_lookahead (role==2) の場合は敵も P1 と同じ探索設定で着手する。
    let p2_lookahead = if enemy_game && !enemy_lookahead {
        LookaheadConfig { policy_only: true, ..lookahead }
    } else {
        lookahead
    };
    let p1_fut = run_player(
        game_id, Player::P1, p1_species, p2_species, stage, lookahead, eval_rule_p1, p1_view,
        client.clone(), p1_rng,
    );
    let p2_fut = run_player(
        game_id, Player::P2, p2_species, p1_species, stage, p2_lookahead, eval_rule_opponent,
        p2_view, client, p2_rng,
    );
    let players = async { tokio::join!(p1_fut, p2_fut) };
    tokio::pin!(players);

    let (p1_res, p2_res) = loop {
        tokio::select! {
            item = inference_rx.recv() => {
                let Some(item) = item else {
                    return false;
                };
                let tx = pending.lock().expect("pending poisoned").remove(&item.request_id);
                if let Some(tx) = tx {
                    let _ = tx.send(item);
                }
            }
            results = &mut players => break results,
        }
    };
    let winner = p1_res.winner.or(p2_res.winner);

    let p1_traj = Trajectory {
        game_id,
        game_index: iter,
        player: Player::P1,
        winner,
        enemy_game,
        items: p1_res.items,
    };
    let p2_traj = Trajectory {
        game_id,
        game_index: iter,
        player: Player::P2,
        winner,
        enemy_game,
        items: p2_res.items,
    };
    if sender.send(RootEnumFromGame::GameHasEnded(p1_traj)).is_err() {
        return false;
    }
    // 評価モードでは P2 (ルール側) の trajectory は学習対象でなく、勝率集計の二重計上を
    // 招くので送らない。学習 (自己対戦) では両者を送る。
    if !eval_rule_opponent && sender.send(RootEnumFromGame::GameHasEnded(p2_traj)).is_err() {
        return false;
    }
    true
}

struct PlayerResult {
    items: Vec<TrajectoryItem>,
    winner: Option<Player>,
}

#[allow(clippy::too_many_arguments)]
async fn run_player<T: ShowdownTrait>(
    game_id: usize,
    player: Player,
    my_species: SpeciesId,
    opp_species: SpeciesId,
    stage: Stage,
    lookahead: LookaheadConfig,
    rule_based: bool,
    mut view: BattleView<T>,
    client: InferenceClient,
    mut rng: ChaCha8Rng,
) -> PlayerResult {
    let _ = stage;
    let mut items: Vec<TrajectoryItem> = Vec::new();
    let mut my_hp_frac: f32 = 1.0;
    let mut opp_hp_frac: f32 = 1.0;
    let mut winner: Option<Player> = None;
    let my_player_num = player.index() as u8 + 1;

    loop {
        let msg = match view.receive().await {
            Ok(m) => m,
            Err(_) => break,
        };
        match msg {
            ReceiveInfo::Request(req) => match req {
                Request::Wait => {}
                Request::TeamPreview => {
                    if view.send(SendInfo::Team(vec![1])).await.is_err() {
                        return PlayerResult { items, winner };
                    }
                }
                Request::Move { moves, .. } => {
                    if rule_based {
                        // 固定ルール: 真の状態から決定論着手。学習サンプルは積まないが、
                        // 評価の対面分類用に初手 (turn1) だけ state+着手を記録する
                        // (target/selection/win_rates はダミー 0)。
                        let send = match view.battle_state() {
                            Some(state) => {
                                let choice = rule_choice(&state, player);
                                if items.is_empty() {
                                    items.push(TrajectoryItem {
                                        game_id,
                                        player,
                                        state: observation_for(&state, player),
                                        target_pi: [0.0; ACTION_DIM],
                                        target_value: 0.0,
                                        selection_pi: [0.0; ACTION_DIM],
                                        chosen_action: action_index(state.party(player), choice) as u8,
                                        win_rates: [0.0; ACTION_DIM],
                                    });
                                }
                                choice_to_sendinfo(choice)
                            }
                            None => {
                                let m = moves
                                    .iter()
                                    .find(|m| !m.disabled)
                                    .and_then(|m| MoveId::from_showdown_id(&m.id));
                                SendInfo::Move(m.map(|m| m.showdown_slot()).unwrap_or(1))
                            }
                        };
                        if view.send(send).await.is_err() {
                            return PlayerResult { items, winner };
                        }
                        continue;
                    }
                    let send = if let Some(state) = view.battle_state() {
                        // Local: 真の状態から lookahead で (training_pi, value) を作り、
                        // 技+交代を含む selection_pi から着手をサンプルする。
                        // policy_only モードでは rollout を廃し policy 単発推論で着手する。
                        let step = decide_step(&client, &lookahead, &state, player, &mut rng).await;
                        items.push(TrajectoryItem {
                            game_id,
                            player,
                            state: observation_for(&state, player),
                            target_pi: step.target_pi,
                            target_value: step.target_value,
                            selection_pi: step.selection_pi,
                            chosen_action: action_index(state.party(player), step.choice) as u8,
                            win_rates: step.win_rates,
                        });
                        choice_to_sendinfo(step.choice)
                    } else {
                        // Showdown: lookahead 非対応。event 由来の観測で 1 回推論する
                        // (技選択のみ。控えは隠匿の None 枠)。request の技リスト順が
                        // そのままスロット順 = 行動 index になる。
                        let mut mask = vec![false; ACTION_DIM];
                        let mut my_move_gids = Vec::new();
                        for (slot, m) in moves.iter().enumerate() {
                            if let Some(id) = MoveId::from_showdown_id(&m.id) {
                                my_move_gids.push(move_gid_of(id));
                                if !m.disabled && slot < ACTION_DIM {
                                    mask[slot] = true;
                                }
                            }
                        }
                        let state = StateForPlayer {
                            my_species_gid: gid_of_species(my_species),
                            opp_species_gid: gid_of_species(opp_species),
                            my_exact_hp_frac: my_hp_frac,
                            opp_quantized_hp_frac: opp_hp_frac,
                            my_move_gids,
                            // Showdown 経路は revealed-only 未対応: 相手側は未知として空。
                            opp_move_gids: Vec::new(),
                            my_bench: vec![None; NUM_BENCH],
                            opp_bench: vec![None; NUM_BENCH],
                            legal_action_mask: mask.clone(),
                        };
                        let inf = client.infer_public(player, state.clone()).await;
                        let chosen_slot = sample_slot_from_pi(&inf.policy, &mask, &mut rng);
                        items.push(TrajectoryItem {
                            game_id,
                            player,
                            state,
                            target_pi: inf.policy,
                            target_value: inf.value,
                            selection_pi: inf.policy,
                            chosen_action: chosen_slot as u8,
                            win_rates: [0.0; ACTION_DIM],
                        });
                        SendInfo::Move((chosen_slot + 1) as u8)
                    };
                    if view.send(send).await.is_err() {
                        return PlayerResult { items, winner };
                    }
                }
                // 強制交代: Local では学習決定ノードとして扱うが、合法手が 1 つ
                // (Stage3b の単一控え等) なら短絡即決し、学習サンプルは記録しない
                // (交代学習の信号は通常ターンの「攻撃 vs 交代」から来る)。本来の選択学習は
                // 3v3 で控えが複数になったときに育つ。Showdown 経路は最初の生存控えへ機械交代。
                Request::ForceSwitch { team } => {
                    if rule_based {
                        let send = match view.battle_state() {
                            Some(state) => choice_to_sendinfo(rule_choice(&state, player)),
                            None => {
                                let slot =
                                    team.iter().position(|m| !m.fainted).unwrap_or(0) + 1;
                                SendInfo::Switch(slot as u8)
                            }
                        };
                        if view.send(send).await.is_err() {
                            return PlayerResult { items, winner };
                        }
                        continue;
                    }
                    let send = if let Some(state) = view.battle_state() {
                        let legal = state.legal_choices(player);
                        if legal.len() == 1 {
                            // 短絡: 唯一の控えへ。サンプルは積まない。
                            choice_to_sendinfo(legal[0])
                        } else {
                            let step =
                                decide_step(&client, &lookahead, &state, player, &mut rng).await;
                            items.push(TrajectoryItem {
                                game_id,
                                player,
                                state: observation_for(&state, player),
                                target_pi: step.target_pi,
                                target_value: step.target_value,
                                selection_pi: step.selection_pi,
                                chosen_action: action_index(state.party(player), step.choice) as u8,
                                win_rates: step.win_rates,
                            });
                            choice_to_sendinfo(step.choice)
                        }
                    } else {
                        let slot = team.iter().position(|m| !m.fainted).unwrap_or(0) + 1;
                        SendInfo::Switch(slot as u8)
                    };
                    if view.send(send).await.is_err() {
                        return PlayerResult { items, winner };
                    }
                }
            },
            ReceiveInfo::Update(update) => {
                for ev in &update.events {
                    match ev {
                        Event::Win { player: name } => {
                            winner = match name.as_str() {
                                "P1" => Some(Player::P1),
                                "P2" => Some(Player::P2),
                                _ => None,
                            };
                            return PlayerResult { items, winner };
                        }
                        Event::Tie => {
                            return PlayerResult { items, winner: None };
                        }
                        Event::Damage {
                            target,
                            hp,
                            max_hp,
                            fainted,
                        } => {
                            if target.player == my_player_num {
                                my_hp_frac = my_exact_hp_frac(*hp, *max_hp, *fainted);
                            } else {
                                opp_hp_frac = opp_quantized_hp_frac(*hp, *max_hp, *fainted);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    PlayerResult { items, winner }
}

/// selection_pi から、真の状態の合法手 (技+交代) に制限して 1 つの `Choice` をサンプルする。
/// 交代手は相対控え index で pi を参照する (env 層の `action_index` に閉じる)。
/// 1 手番の決定結果。着手と、学習サンプルに積む (target/selection/value/win_rates)。
struct StepResult {
    choice: Choice,
    target_pi: [f32; ACTION_DIM],
    target_value: f32,
    selection_pi: [f32; ACTION_DIM],
    win_rates: [f32; ACTION_DIM],
}

/// 真の状態から 1 手を決める。通常は lookahead、`policy_only` 時は policy net を 1 回だけ
/// 推論して確率サンプルする (rollout なし)。後者は checkpoint 同士の高速強さ判定用。
async fn decide_step(
    client: &InferenceClient,
    lookahead: &LookaheadConfig,
    state: &BattleState,
    player: Player,
    rng: &mut ChaCha8Rng,
) -> StepResult {
    if lookahead.policy_only {
        let out = client.infer(observation_for(state, player), player).await;
        let choice = sample_choice_from_pi(&out.policy, state, player, rng);
        StepResult {
            choice,
            target_pi: out.policy,
            target_value: out.value,
            selection_pi: out.policy,
            win_rates: [0.0; ACTION_DIM],
        }
    } else {
        let seed: u64 = rng.r#gen();
        let result = run_lookahead(client, *state, player, seed, lookahead).await;
        let choice = sample_choice_from_pi(&result.selection_pi, state, player, rng);
        StepResult {
            choice,
            target_pi: result.training_pi,
            target_value: result.value,
            selection_pi: result.selection_pi,
            win_rates: result.win_rates,
        }
    }
}

fn sample_choice_from_pi(
    pi: &[f32; ACTION_DIM],
    state: &BattleState,
    player: Player,
    rng: &mut ChaCha8Rng,
) -> Choice {
    let party = state.party(player);
    let legal = state.legal_choices(player);
    let mut total = 0.0f32;
    for c in &legal {
        total += pi[action_index(party, *c)].max(0.0);
    }
    if total <= 0.0 {
        return legal[0];
    }
    let r: f32 = rng.gen_range(0.0..total);
    let mut acc = 0.0f32;
    for c in &legal {
        acc += pi[action_index(party, *c)].max(0.0);
        if r < acc {
            return *c;
        }
    }
    *legal.last().unwrap()
}

/// `Choice` を Showdown の送信コマンドへ。技は技スロット、交代は絶対パーティ index+1。
fn choice_to_sendinfo(choice: Choice) -> SendInfo {
    match choice {
        Choice::Move(m) => SendInfo::Move(m.showdown_slot()),
        Choice::Switch(abs) => SendInfo::Switch((abs + 1) as u8),
    }
}

/// 種族のグローバル ID (Showdown 経路用、SpeciesId 経由)。
fn gid_of_species(species: SpeciesId) -> u16 {
    species_meta(species.name())
        .expect("scenario species must be in the global id table")
        .id
}

/// 確率分布 `pi` から legal な技スロットに制限して 1 つ選ぶ (Showdown 経路用)。
fn sample_slot_from_pi(
    pi: &[f32; ACTION_DIM],
    legal: &[bool],
    rng: &mut ChaCha8Rng,
) -> usize {
    let slots: Vec<usize> = (0..legal.len().min(ACTION_DIM)).filter(|&i| legal[i]).collect();
    let Some(&first) = slots.first() else {
        return 0;
    };
    let total: f32 = slots.iter().map(|&i| pi[i].max(0.0)).sum();
    if total <= 0.0 {
        return first;
    }
    let r: f32 = rng.gen_range(0.0..total);
    let mut acc = 0.0f32;
    for &i in &slots {
        acc += pi[i].max(0.0);
        if r < acc {
            return i;
        }
    }
    *slots.last().unwrap()
}

/// 自分のポケモンの HP 割合。Showdown の secret 行 (実数 HP) から正確に求める。
fn my_exact_hp_frac(hp: u32, max_hp: u32, fainted: bool) -> f32 {
    if fainted || max_hp == 0 {
        return 0.0;
    }
    (hp as f32 / max_hp as f32).clamp(0.0, 1.0)
}

/// 相手のポケモンの HP 割合。Showdown が相手に見せる整数パーセントを再現する。
fn opp_quantized_hp_frac(hp: u32, max_hp: u32, fainted: bool) -> f32 {
    if fainted || max_hp == 0 || hp == 0 {
        return 0.0;
    }
    let pct = (100 * hp + max_hp - 1) / max_hp; // ceil(100 * hp / max_hp)
    let pct = if pct >= 100 && hp < max_hp { 99 } else { pct };
    pct as f32 / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }

    #[test]
    fn stage3b_pick_config_is_always_cross_team() {
        // Stage3b は必ず Team1 vs Team2。多数の (game_id, iter) で同チーム対戦が出ないこと、
        // かつ 8 通りの構成がすべて現れることを確認する。
        let mut seen = std::collections::HashSet::new();
        for game_id in 0..50 {
            for iter in 1..200u32 {
                let c = pick_config(game_id, iter, Stage::Stage3b);
                assert_ne!(
                    c.p1_team_text, c.p2_team_text,
                    "pick_config must pair Team1 vs Team2 (game_id={game_id}, iter={iter})"
                );
                seen.insert((c.p1_team_text.clone(), c.p2_team_text.clone(),
                             c.p1_species.index(), c.p2_species.index()));
            }
        }
        assert_eq!(seen.len(), 8, "all 8 cross-team configs should appear, got {}", seen.len());
    }

    #[test]
    fn my_exact_hp_frac_is_exact_and_zero_when_fainted() {
        approx(my_exact_hp_frac(175, 175, false), 1.0);
        approx(my_exact_hp_frac(138, 175, false), 138.0 / 175.0);
        approx(my_exact_hp_frac(0, 175, true), 0.0);
        approx(my_exact_hp_frac(1, 175, true), 0.0);
    }

    #[test]
    fn opp_quantized_hp_frac_matches_showdown_percentages() {
        approx(opp_quantized_hp_frac(175, 175, false), 1.0);
        approx(opp_quantized_hp_frac(1, 200, false), 0.01);
        approx(opp_quantized_hp_frac(174, 175, false), 0.99);
        approx(opp_quantized_hp_frac(138, 175, false), 0.79);
        approx(opp_quantized_hp_frac(0, 175, true), 0.0);
        approx(opp_quantized_hp_frac(0, 175, false), 0.0);
    }
}
