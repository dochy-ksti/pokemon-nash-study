//! Pokemon Showdown / Showdown 互換シミュレータと通信するための trait と
//! その `pokemon-showdown simulate-battle` 実装。
//!
//! プロトコル概要:
//! - シミュレータからのメッセージは「ブロック」単位で来る。
//!   最初の 1 行がブロック種別 (`update` または `sideupdate`)、空行で区切られる。
//!   `sideupdate` の場合は次の 1 行に `p1` / `p2` が入る。
//!   それ以降の行が中身 (`|...`) で、`|request|{json}` か通常の update item。
//! - こちら側は `>start ...`, `>player p1 ...`, `>p1 move N` などを書き込む。

use std::path::PathBuf;
use std::process::Stdio;

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::mpsc;

use poke_sho_rust::event::{Event, PokemonRef};
use poke_sho_rust::battle::Player;

use crate::protocol;

/// ShowdownTrait — シミュレータ側との 1 プレイヤー分の窓口。
///
/// 実装ごとに `receive` は対応する side 向けの update / request を返し、
/// `send` はその side からの行動を送る。
pub trait ShowdownTrait {
    fn receive(
        &mut self,
    ) -> impl std::future::Future<Output = Result<ReceiveInfo, ShowdownError>> + Send;

    fn send(
        &mut self,
        send_info: SendInfo,
    ) -> impl std::future::Future<Output = Result<(), ShowdownError>> + Send;

    /// 直近に提示された手番開始時点の真の `BattleState`。
    ///
    /// lookahead (Monte-Carlo rollout) はこの真値を起点に局面を展開する。
    /// Showdown subprocess バックエンドは内部状態を持たないため `None` を返し、
    /// in-process な `LocalShowdown` のみが実値を返す。
    fn battle_state(&self) -> Option<poke_sho_rust::battle::BattleState> {
        None
    }
}

#[derive(Debug, Error)]
pub enum ShowdownError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("simulator channel closed")]
    Closed,
    #[error("invalid update: {0}")]
    InvalidUpdate(String),
    #[error("subprocess exited unexpectedly: {0}")]
    Subprocess(String),
}

#[derive(Debug, Clone)]
pub enum ReceiveInfo {
    Request(Request),
    Update(Update),
}

/// 選択可能な行動を表す型付き request。Phase1 で実際に来る種別のみをモデル化する。
/// force switch は phase4(交代導入)で追加予定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// team preview。応答は team order(phase1 は 1 体なので `team 1`)。
    TeamPreview,
    /// アクティブ 1 体の技選択。`moves` は選択可否付きの技スロット列、
    /// `team` は自軍パーティ全員の現在 HP(request JSON の `side.pokemon` 由来)。
    /// 相手の HP は request には含まれないため、自軍のみを持つ。
    Move {
        moves: Vec<MoveOption>,
        team: Vec<TeamMember>,
    },
    /// 瀕死後の強制交代。交代手だけが合法で、`team` は控えの状態(交代先選択用)。
    /// Showdown `|request|{"forceSwitch":[true],"side":{...}}` 由来。
    ForceSwitch {
        team: Vec<TeamMember>,
    },
    /// この手番で選ぶものがない(相手の決定待ち等)。
    Wait,
}

/// 1 つの技スロットの選択可否。`id` は Showdown 正規化 id (`toID`)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveOption {
    pub id: String,
    pub disabled: bool,
}

/// 自軍パーティ 1 体分の状態。`mon` は ident 由来、HP は実数(自軍なので
/// パーセント丸めではなく `cur/max` がそのまま得られる)。瀕死は `0 fnt` →
/// `hp = 0, max_hp = 0, fainted = true`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeamMember {
    pub mon: PokemonRef,
    pub hp: u32,
    pub max_hp: u32,
    pub fainted: bool,
}

/// 1 ブロック分の battle 更新を共通 Event 列として表す。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Update {
    pub events: Vec<Event>,
}

#[derive(Debug, Clone)]
pub enum SendInfo {
    /// 1..=4。Phase1 では 1 or 2。
    Move(u8),
    /// 交代。`switch N`(N は 1 始まりのパーティ位置)。通常ターン・強制交代の双方で使う。
    Switch(u8),
    /// teamPreview への応答。`[1]` なら `team 1`。
    Team(Vec<u8>),
    /// `default` (強制スイッチや forceSwitch などへのフォールバック)。
    Default,
}

/// `pokemon-showdown simulate-battle` を使う `ShowdownTrait` 実装。
///
/// `SimulatorDriver` (シミュレータ全体) と内部チャネルで繋がっており、
/// `receive` で自分の side 向けメッセージを待ち受け、`send` で行動を送る。
pub struct SimulatorShowdown {
    side: Player,
    rx: mpsc::UnboundedReceiver<ReceiveInfo>,
    tx: mpsc::UnboundedSender<(Player, SendInfo)>,
}

impl ShowdownTrait for SimulatorShowdown {
    async fn receive(&mut self) -> Result<ReceiveInfo, ShowdownError> {
        self.rx.recv().await.ok_or(ShowdownError::Closed)
    }

    async fn send(&mut self, send_info: SendInfo) -> Result<(), ShowdownError> {
        self.tx
            .send((self.side, send_info))
            .map_err(|_| ShowdownError::Closed)
    }
}

/// Showdown からのデータを解析しつつ、生の `ShowdownTrait` をラップする。
/// Phase1 段階では「受信ループを回して必要に応じて状態を抽出する」
/// 程度の薄いラッパだが、将来的に相手ポケモンの情報トラッキング等を担う。
pub struct BattleView<T: ShowdownTrait> {
    showdown: T,
}

impl<T: ShowdownTrait> BattleView<T> {
    pub fn new(showdown: T) -> Self {
        Self { showdown }
    }

    pub async fn receive(&mut self) -> Result<ReceiveInfo, ShowdownError> {
        self.showdown.receive().await
    }

    pub async fn send(&mut self, send_info: SendInfo) -> Result<(), ShowdownError> {
        self.showdown.send(send_info).await
    }

    /// 直近の手番開始時点の真の `BattleState` (lookahead 用)。
    pub fn battle_state(&self) -> Option<poke_sho_rust::battle::BattleState> {
        self.showdown.battle_state()
    }
}

/// シミュレータ全体 (p1 + p2 + バックグラウンド subprocess) を保持する。
pub struct SimulatorDriver {
    pub p1: BattleView<SimulatorShowdown>,
    pub p2: BattleView<SimulatorShowdown>,
    _supervisor: SimulatorSupervisor,
}

impl SimulatorDriver {
    /// バックグラウンドタスクを止めてプロセスを終了させる。
    pub async fn shutdown(self) {
        drop(self.p1);
        drop(self.p2);
        let _ = self._supervisor.shutdown().await;
    }
}

struct SimulatorSupervisor {
    child: Child,
}

impl SimulatorSupervisor {
    async fn shutdown(mut self) -> Result<(), ShowdownError> {
        // stdin を閉じれば simulate-battle は終了する。
        // 上流タスクが stdin を保持しているので、kill で確実に止める。
        let _ = self.child.start_kill();
        let _ = self.child.wait().await;
        Ok(())
    }
}

/// `pokemon-showdown simulate-battle` を起動して `SimulatorDriver` を返す。
///
/// `team1`, `team2` は Showdown のチームテキスト形式 (パック前) を受け取り、
/// 内部で `pack-team` を通して packed text に変換する。
/// `format_id` には `gen9customgame` などを渡す。
pub async fn create_simulator_game(
    showdown_dir: PathBuf,
    team1: String,
    team2: String,
    format_id: &str,
    seed: [u32; 4],
) -> Result<SimulatorDriver, ShowdownError> {
    let packed1 = pack_team(&showdown_dir, &team1).await?;
    let packed2 = pack_team(&showdown_dir, &team2).await?;

    let mut child = Command::new(showdown_dir.join("pokemon-showdown"))
        .arg("simulate-battle")
        .current_dir(&showdown_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| ShowdownError::Subprocess("stdin missing".into()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ShowdownError::Subprocess("stdout missing".into()))?;

    let (p1_tx, p1_rx) = mpsc::unbounded_channel();
    let (p2_tx, p2_rx) = mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<(Player, SendInfo)>();

    // 起動コマンドを cmd チャネルに先回しで詰める。
    let start_payload = format!(
        r#">start {{"formatid":"{format_id}","seed":[{},{},{},{}]}}"#,
        seed[0], seed[1], seed[2], seed[3]
    );
    let p1_payload = format!(r#">player p1 {{"name":"P1","team":"{}"}}"#, packed1);
    let p2_payload = format!(r#">player p2 {{"name":"P2","team":"{}"}}"#, packed2);

    // writer タスク: subprocess の stdin に書き続ける
    tokio::spawn(writer_task(
        stdin,
        cmd_rx,
        vec![start_payload, p1_payload, p2_payload],
    ));

    // reader タスク: stdout を読んで p1/p2 にディスパッチ
    tokio::spawn(reader_task(stdout, p1_tx, p2_tx));

    let p1 = BattleView::new(SimulatorShowdown {
        side: Player::P1,
        rx: p1_rx,
        tx: cmd_tx.clone(),
    });
    let p2 = BattleView::new(SimulatorShowdown {
        side: Player::P2,
        rx: p2_rx,
        tx: cmd_tx,
    });

    Ok(SimulatorDriver {
        p1,
        p2,
        _supervisor: SimulatorSupervisor { child },
    })
}

async fn pack_team(showdown_dir: &PathBuf, text: &str) -> Result<String, ShowdownError> {
    let mut child = Command::new(showdown_dir.join("pokemon-showdown"))
        .arg("pack-team")
        .current_dir(showdown_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(text.as_bytes()).await?;
        stdin.shutdown().await?;
    }
    let output = child.wait_with_output().await?;
    if !output.status.success() {
        return Err(ShowdownError::Subprocess(format!(
            "pack-team failed: {}",
            output.status
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

async fn writer_task(
    mut stdin: ChildStdin,
    mut cmd_rx: mpsc::UnboundedReceiver<(Player, SendInfo)>,
    initial: Vec<String>,
) {
    for line in initial {
        if stdin.write_all(line.as_bytes()).await.is_err() {
            return;
        }
        if stdin.write_all(b"\n").await.is_err() {
            return;
        }
    }
    if stdin.flush().await.is_err() {
        return;
    }
    while let Some((player, info)) = cmd_rx.recv().await {
        let side = match player {
            Player::P1 => "p1",
            Player::P2 => "p2",
        };
        let line = match info {
            SendInfo::Move(n) => format!(">{side} move {n}\n"),
            SendInfo::Switch(n) => format!(">{side} switch {n}\n"),
            SendInfo::Team(order) => {
                let csv = order
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!(">{side} team {csv}\n")
            }
            SendInfo::Default => format!(">{side} default\n"),
        };
        if stdin.write_all(line.as_bytes()).await.is_err() {
            return;
        }
        if stdin.flush().await.is_err() {
            return;
        }
    }
    let _ = stdin.shutdown().await;
}

async fn reader_task(
    stdout: tokio::process::ChildStdout,
    p1_tx: mpsc::UnboundedSender<ReceiveInfo>,
    p2_tx: mpsc::UnboundedSender<ReceiveInfo>,
) {
    let mut reader = BufReader::new(stdout).lines();
    let mut section: Option<String> = None;
    let mut side_label: Option<String> = None;
    let mut buf: Vec<String> = Vec::new();

    loop {
        let next = match reader.next_line().await {
            Ok(line) => line,
            Err(_) => return,
        };
        let Some(line) = next else { return };

        if line.is_empty() {
            flush_block(&section, &side_label, &buf, &p1_tx, &p2_tx);
            section = None;
            side_label = None;
            buf.clear();
            continue;
        }

        if section.is_none() {
            section = Some(line);
            continue;
        }
        if section.as_deref() == Some("sideupdate") && side_label.is_none() {
            side_label = Some(line);
            continue;
        }
        buf.push(line);
    }
}

fn flush_block(
    section: &Option<String>,
    side_label: &Option<String>,
    buf: &[String],
    p1_tx: &mpsc::UnboundedSender<ReceiveInfo>,
    p2_tx: &mpsc::UnboundedSender<ReceiveInfo>,
) {
    let Some(section) = section else { return };

    match section.as_str() {
        "sideupdate" => {
            let Some(side) = side_label else { return };
            for line in buf {
                if let Some(rest) = line.strip_prefix("|request|") {
                    let info = ReceiveInfo::Request(protocol::parse_request_json(rest));
                    match side.as_str() {
                        "p1" => {
                            let _ = p1_tx.send(info);
                        }
                        "p2" => {
                            let _ = p2_tx.send(info);
                        }
                        _ => {}
                    }
                }
            }
        }
        "update" => {
            let events = protocol::parse_update_block(buf);
            if events.is_empty() {
                return;
            }
            let update = Update { events };
            // update は両 side に同じものを送る。
            let _ = p1_tx.send(ReceiveInfo::Update(update.clone()));
            let _ = p2_tx.send(ReceiveInfo::Update(update));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn showdown_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("pokemon-showdown")
    }

    fn cloyster_team() -> String {
        use poke_sho_rust::scenario::{SpeciesId, Stage};
        SpeciesId::Cloyster.team_text(Stage::Stage3a).to_string()
    }

    #[tokio::test]
    async fn run_scenario_random_battle() {
        let dir = showdown_dir();
        if !dir.join("pokemon-showdown").exists() {
            eprintln!("pokemon-showdown CLI not found, skipping");
            return;
        }
        let mut driver = create_simulator_game(
            dir,
            cloyster_team(),
            cloyster_team(),
            "gen9customgame",
            [1, 2, 3, 4],
        )
        .await
        .expect("create simulator");

        // ランダム同士で戦って勝者が出ることを確認する。
        let mut winner: Option<String> = None;
        let mut p1_done = false;
        let mut p2_done = false;

        let timeout = tokio::time::Duration::from_secs(30);
        let result = tokio::time::timeout(timeout, async {
            loop {
                tokio::select! {
                    msg = driver.p1.receive(), if !p1_done => {
                        match msg {
                            Ok(ReceiveInfo::Request(req)) => {
                                if let Some(action) = pick_action(&req) {
                                    driver.p1.send(action).await.unwrap();
                                }
                            }
                            Ok(ReceiveInfo::Update(update)) => {
                                if let Some(w) = win_player(&update.events) { winner = Some(w); }
                            }
                            Err(_) => { p1_done = true; }
                        }
                    }
                    msg = driver.p2.receive(), if !p2_done => {
                        match msg {
                            Ok(ReceiveInfo::Request(req)) => {
                                if let Some(action) = pick_action(&req) {
                                    driver.p2.send(action).await.unwrap();
                                }
                            }
                            Ok(ReceiveInfo::Update(update)) => {
                                if let Some(w) = win_player(&update.events) { winner = Some(w); }
                            }
                            Err(_) => { p2_done = true; }
                        }
                    }
                }
                if winner.is_some() {
                    break;
                }
            }
        })
        .await;
        assert!(result.is_ok(), "simulator timed out");
        assert!(winner.is_some(), "no winner produced");
        driver.shutdown().await;
    }

    fn pick_action(req: &Request) -> Option<SendInfo> {
        match req {
            Request::TeamPreview => Some(SendInfo::Team(vec![1])),
            Request::Wait => None,
            Request::Move { moves, .. } => moves
                .iter()
                .position(|m| !m.disabled)
                .map(|i| SendInfo::Move((i + 1) as u8)),
            // 瀕死後は最初の生存控えへ交代する。
            Request::ForceSwitch { team } => team
                .iter()
                .position(|m| !m.fainted)
                .map(|i| SendInfo::Switch((i + 1) as u8)),
        }
    }

    fn win_player(events: &[Event]) -> Option<String> {
        events.iter().find_map(|e| match e {
            Event::Win { player } => Some(player.clone()),
            _ => None,
        })
    }
}
