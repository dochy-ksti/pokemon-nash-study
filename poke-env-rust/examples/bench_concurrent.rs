//! Pokemon Showdown subprocess の並列スループット計測。
//!
//! 使い方:
//! cargo run --release --example bench_concurrent -- <num_games>

use poke_env_rust::observation::Event;
use poke_env_rust::showdown_trait::{
    BattleView, ReceiveInfo, Request, SendInfo, SimulatorShowdown, ShowdownError,
    create_simulator_game,
};
use std::path::PathBuf;
use std::time::Instant;

fn cloyster_team() -> String {
    use poke_sho_rust::scenario::{SpeciesId, Stage};
    SpeciesId::Cloyster.team_text(Stage::Stage3a).to_string()
}

fn showdown_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("pokemon-showdown")
}

fn pick_action(req: &Request) -> Option<SendInfo> {
    match req {
        Request::TeamPreview => Some(SendInfo::Team(vec![1])),
        Request::Wait => None,
        Request::Move { moves, .. } => moves
            .iter()
            .position(|m| !m.disabled)
            .map(|i| SendInfo::Move((i + 1) as u8)),
        Request::ForceSwitch { team } => team
            .iter()
            .position(|m| !m.fainted)
            .map(|i| SendInfo::Switch((i + 1) as u8)),
    }
}

async fn drive_player(mut view: BattleView<SimulatorShowdown>) -> Result<(), ShowdownError> {
    loop {
        match view.receive().await? {
            ReceiveInfo::Request(req) => {
                if let Some(action) = pick_action(&req) {
                    view.send(action).await?;
                }
            }
            ReceiveInfo::Update(update) => {
                for ev in &update.events {
                    if matches!(ev, Event::Win { .. } | Event::Tie) {
                        return Ok(());
                    }
                }
            }
        }
    }
}

async fn one_game(game_id: u32) -> (f64, f64) {
    let t_spawn = Instant::now();
    let driver = create_simulator_game(
        showdown_dir(),
        cloyster_team(),
        cloyster_team(),
        "gen9customgame",
        [game_id, game_id + 1, game_id + 2, game_id + 3],
    )
    .await
    .expect("create");
    let spawn_ms = t_spawn.elapsed().as_secs_f64() * 1000.0;

    let t_play = Instant::now();
    let p1 = tokio::spawn(drive_player(driver.p1));
    let p2 = tokio::spawn(drive_player(driver.p2));
    let _ = p1.await;
    let _ = p2.await;
    let play_ms = t_play.elapsed().as_secs_f64() * 1000.0;
    (spawn_ms, play_ms)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    eprintln!("=== Sequential 1 game (warmup) ===");
    let (sp, pl) = one_game(0).await;
    eprintln!("spawn={:.0}ms play={:.0}ms total={:.0}ms", sp, pl, sp + pl);

    eprintln!("\n=== Concurrent {} games ===", n);
    let t_total = Instant::now();
    let mut handles = Vec::new();
    for i in 0..n {
        handles.push(tokio::spawn(async move {
            let (sp, pl) = one_game(1000 + i as u32 * 4).await;
            (i, sp, pl)
        }));
    }
    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    results.sort_by_key(|r| r.0);
    for (i, sp, pl) in &results {
        eprintln!("game[{:2}] spawn={:6.0}ms play={:6.0}ms", i, sp, pl);
    }
    let spawn_avg: f64 = results.iter().map(|r| r.1).sum::<f64>() / results.len() as f64;
    let play_avg: f64 = results.iter().map(|r| r.2).sum::<f64>() / results.len() as f64;
    eprintln!(
        "\ntotal wall={:.0}ms | avg spawn={:.0}ms avg play={:.0}ms | throughput={:.2} games/s",
        total_ms,
        spawn_avg,
        play_avg,
        n as f64 / (total_ms / 1000.0)
    );
}
