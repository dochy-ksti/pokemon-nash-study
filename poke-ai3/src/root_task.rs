use poke_env_rust::observation::{Player, StateForPlayer};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::async_executor::{ACTION_DIM, ExecutorEnum};

#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
pub struct PlayerObservation {
    pub game_id: usize,
    /// この game_id スロットで何ゲーム目か (1 始まり)。同一スロットは前ゲーム終了後に
    /// 次ゲームを逐次開始するので、(game_id, game_index) が 1 バトルを一意に識別する。
    /// Python 側の敵混合学習が「新ゲーム開始」を検知して敵を σ 配分で割り当てる鍵。
    pub game_index: u32,
    pub player: Player,
    /// このゲーム内で一意な推論リクエスト ID。応答 (`InferencedDataItem`) が
    /// 同じ ID を返し、game_task 側で待機中の oneshot へルーティングする。
    pub request_id: u64,
    /// 実観測なら `Some`、empty (ウィンドウ穴埋め用の計上専用ダミー) なら `None`。
    /// empty は GPU forward に載せず、Python は request_id だけをエコーバックして
    /// ack を返す。これによりウィンドウ常時 W 本を維持し max_batch_size ゲートのデッドロックを防ぐ。
    pub state: Option<StateForPlayer>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
pub struct PlayerObservations {
    pub vec: Vec<PlayerObservation>,
}

/// lookahead が算出した 1 局面分の学習サンプル。観測 (state) に対する
/// policy 教師 (`target_pi`) と value 教師 (`target_value`) を持つ。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrajectoryItem {
    pub game_id: usize,
    pub player: Player,
    pub state: StateForPlayer,
    pub target_pi: [f32; ACTION_DIM],
    pub target_value: f32,
    /// 実際に着手をサンプルした元の分布 (local は lookahead の selection_pi、
    /// showdown は net policy)。target_pi (学習用) との差を診断で見るために記録。
    pub selection_pi: [f32; ACTION_DIM],
    /// 実際に選ばれた行動の ACTION_DIM index (技は技スロット、交代は MAX_MOVE_SLOTS 以降)。
    pub chosen_action: u8,
    /// lookahead の各合法手 rollout 平均勝率 (root 視点 0..1)。無差別局面の診断用。
    /// showdown 経路 (lookahead 非対応) では全 0。
    pub win_rates: [f32; ACTION_DIM],
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Trajectory {
    pub game_id: usize,
    /// この game_id スロットで何ゲーム目か (1 始まり)。観測と同じ値を持ち、Python 側が
    /// (game_id, game_index) で敵割り当てを引いて敵別勝率を集計する。
    pub game_index: u32,
    /// この trajectory を生成した player。1 実バトルは P1/P2 の 2 本を生むため、
    /// 評価時の実試合数集計では P1 側だけを数えて二重計上を避ける。
    pub player: Player,
    pub winner: Option<Player>,
    /// この trajectory が「敵ゲーム」(P2 が凍結した過去 checkpoint) で生成されたか。
    /// 敵ゲームでは P2 の手は学習教師にしない (build_examples が P1 のみ採用)。
    /// 自己対戦は false (P1/P2 両方学習)。役割はゲーム開始時に役割テーブルから読む。
    pub enemy_game: bool,
    pub items: Vec<TrajectoryItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Trajectories {
    pub vec: Vec<Trajectory>,
}

pub(crate) enum RootEnumFromGame {
    Data(PlayerObservation),
    GameHasEnded(Trajectory),
}

pub(crate) struct RootTask {
    pub(crate) root_receiver: UnboundedReceiver<RootEnumFromGame>,
    pub(crate) sender_to_executor: UnboundedSender<ExecutorEnum>,
    pub(crate) trajectories: Vec<Trajectory>,
    pub(crate) max_batch_size: usize,
    pub(crate) trajectories_threshold: usize,
    /// 推論バッチサイズ計測 (実際に flush した observation_batch のサイズ統計)。
    pub(crate) batch_stats: BatchStats,
}

struct RootTiming {
    enabled: bool,
    started_at: Instant,
    last_batch_at: Option<Instant>,
    encode_time: Duration,
    batch_interval_time: Duration,
    batches: u64,
    queued_after_encode: u64,
    max_queued_after_encode: usize,
}

impl RootTiming {
    fn from_env() -> Self {
        Self {
            enabled: std::env::var_os("POKE_AI3_ROOT_TIMING").is_some(),
            started_at: Instant::now(),
            last_batch_at: None,
            encode_time: Duration::ZERO,
            batch_interval_time: Duration::ZERO,
            batches: 0,
            queued_after_encode: 0,
            max_queued_after_encode: 0,
        }
    }

    fn record_batch(&mut self, encode_time: Duration, queued_after_encode: usize) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        if let Some(last_batch_at) = self.last_batch_at {
            self.batch_interval_time += now.duration_since(last_batch_at);
        }
        self.last_batch_at = Some(now);
        self.encode_time += encode_time;
        self.batches += 1;
        self.queued_after_encode += queued_after_encode as u64;
        self.max_queued_after_encode = self.max_queued_after_encode.max(queued_after_encode);
        if self.batches % 5000 == 0 {
            self.report();
        }
    }

    fn report(&self) {
        if !self.enabled || self.batches == 0 {
            return;
        }
        let wall = self.started_at.elapsed();
        let avg_interval = if self.batches > 1 {
            self.batch_interval_time.as_secs_f64() / (self.batches - 1) as f64
        } else {
            0.0
        };
        eprintln!(
            "root_timing wall_s={:.3} batches={} encode_s={:.3} encode_ratio={:.4} \
             avg_encode_us={:.2} avg_interval_us={:.2} avg_queued_after_encode={:.2} \
             max_queued_after_encode={}",
            wall.as_secs_f64(),
            self.batches,
            self.encode_time.as_secs_f64(),
            self.encode_time.as_secs_f64() / wall.as_secs_f64(),
            self.encode_time.as_secs_f64() * 1_000_000.0 / self.batches as f64,
            avg_interval * 1_000_000.0,
            self.queued_after_encode as f64 / self.batches as f64,
            self.max_queued_after_encode,
        );
    }
}

/// 実バッチサイズ (send_batch で flush した observation_batch.len()) の統計。
/// max_batch_size (上限キャップ) に対し実際どこまで太れているかを測る。
#[derive(Default)]
pub(crate) struct BatchStats {
    batches: u64,
    observations: u64,
    empties: u64,
    max: usize,
    /// 2 の冪バケットのヒストグラム: [1, 2-4, 5-8, 9-16, 17-32, 33-64, 65-128,
    /// 129-256, 257-512, 513-1024, 1025+]。
    hist: [u64; 11],
}

impl BatchStats {
    fn record(&mut self, size: usize, empties: usize) {
        self.batches += 1;
        self.observations += size as u64;
        self.empties += empties as u64;
        if size > self.max {
            self.max = size;
        }
        let bucket = match size {
            0..=1 => 0,
            2..=4 => 1,
            5..=8 => 2,
            9..=16 => 3,
            17..=32 => 4,
            33..=64 => 5,
            65..=128 => 6,
            129..=256 => 7,
            257..=512 => 8,
            513..=1024 => 9,
            _ => 10,
        };
        self.hist[bucket] += 1;
    }

    fn report(&self, max_batch_size: usize) {
        if self.batches == 0 {
            return;
        }
        let avg = self.observations as f64 / self.batches as f64;
        let total = self.observations + self.empties;
        let empty_ratio = if total > 0 {
            self.empties as f64 / total as f64
        } else {
            0.0
        };
        eprintln!(
            "batch_stats batches={} real={} empty={} empty_ratio={:.3} avg_real={:.1} max={} cap={} hist[1,2-4,5-8,9-16,17-32,33-64,65-128,129-256,257-512,513-1024,1025+]={:?}",
            self.batches,
            self.observations,
            self.empties,
            empty_ratio,
            avg,
            self.max,
            max_batch_size,
            self.hist
        );
    }
}

impl RootTask {
    pub(crate) async fn run(mut self) {
        let mut observation_batch = Vec::with_capacity(self.max_batch_size);
        let timing = Arc::new(Mutex::new(RootTiming::from_env()));
        loop {
            // early-flush はしない。閾値に達するまで GPU を駆動せず、バッチ密度を保つ。
            // キューが空でも flush せず、次の到着を待つだけ。flush は閾値到達か終了時のみ。
            //
            // 注意: max_batch_size が同時に in-flight になり得る observation 数を超えると、永遠に
            // 閾値へ到達せず deadlock する。各ゲームは次の推論結果を待ってから次の observation
            // を送るため、pending は概ね「進行中ゲーム数」で頭打ちになる。max_batch_size は必ず
            // その上限より小さく設定すること。
            let msg = self.root_receiver.recv().await;
            match msg {
                Some(RootEnumFromGame::Data(item)) => {
                    observation_batch.push(item);
                    if observation_batch.len() >= self.max_batch_size {
                        self.send_batch(&mut observation_batch, &timing, true);
                    }
                }
                Some(RootEnumFromGame::GameHasEnded(trajectory)) => {
                    self.trajectories.push(trajectory);
                    if self.trajectories.len() >= self.trajectories_threshold {
                        self.send_trajectories();
                    }
                }
                None => break,
            }
        }
        if !observation_batch.is_empty() {
            // 最終 flush は spawn せず同期実行 (run 終了までに encode 完了を保証)。
            self.send_batch(&mut observation_batch, &timing, false);
        }
        if !self.trajectories.is_empty() {
            self.send_trajectories();
        }
        // 終了時に最終集計を出す。通常バッチは in-flight な encode が残り得るが、
        // 最終 flush は同期 (spawn しない) 経路を通すため、ここまでで全 encode 完了済み。
        self.batch_stats.report(self.max_batch_size);
        timing.lock().expect("root timing poisoned").report();
    }

    /// `offload` が true なら encode+pack を `spawn_blocking` へ隔離し、run ループの
    /// 次バッチ形成と並行させる (通常経路)。最終 flush は false で同期実行し、run
    /// 終了までに encode 完了を保証する。
    fn send_batch(
        &mut self,
        observation_batch: &mut Vec<PlayerObservation>,
        timing: &Arc<Mutex<RootTiming>>,
        offload: bool,
    ) {
        let vec = std::mem::take(observation_batch);
        // GPU バッチサイズ = real (state=Some) のみ。empty は計上専用で GPU に載らない。
        let reals = vec.iter().filter(|o| o.state.is_some()).count();
        let empties = vec.len() - reals;
        self.batch_stats.record(reals, empties);
        // 一定間隔で途中経過を出す (最終集計は run の末尾で出す)。
        if self.batch_stats.batches % 50 == 0 {
            self.batch_stats.report(self.max_batch_size);
        }
        // queued_after_encode は「encode へ投入した時点で root_receiver に滞留している
        // 観測数」として読む。直列版は encode 直後、async 版は spawn 時点の値。
        let queued = self.root_receiver.len();
        let sender = self.sender_to_executor.clone();
        let timing = Arc::clone(timing);
        // 純 CPU の encode+pack を closure に閉じ込める。通常はブロッキングプールへ
        // 隔離して run ループの次バッチ形成と並行させ、最終 flush のみ同期実行する。
        let encode_and_send = move || {
            let encode_started_at = Instant::now();
            let packed = crate::packed::pack_batch(crate::obs_encode::encode_batch(&vec));
            timing
                .lock()
                .expect("root timing poisoned")
                .record_batch(encode_started_at.elapsed(), queued);
            let _ = sender.send(ExecutorEnum::Observations(packed));
        };
        if offload {
            tokio::task::spawn_blocking(encode_and_send);
        } else {
            encode_and_send();
        }
    }

    fn send_trajectories(&mut self) {
        let vec = std::mem::take(&mut self.trajectories);
        let _ = self
            .sender_to_executor
            .send(ExecutorEnum::Trajectories(Trajectories { vec }));
    }
}
