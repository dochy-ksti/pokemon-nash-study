use crate::error::ExecutorError;
use crate::game_task::{GameReceiver, game_start};
pub use crate::root_task::{PlayerObservation, PlayerObservations};
use crate::packed::PackedBatch;
use crate::root_task::{RootTask, Trajectories};
use poke_env_rust::observation::{Player, Stage};
use poke_env_rust::lookahead::LookaheadConfig;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use tokio::runtime::Runtime;
use tokio::sync::mpsc::{
    UnboundedReceiver, UnboundedSender, error::TryRecvError, unbounded_channel,
};

/// 行動空間の次元 (技 + 控えへの交代枠)。env 層の `ACTION_DIM` と常に一致させる。
pub const ACTION_DIM: usize = poke_env_rust::observation::ACTION_DIM;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// `poke-sho-rust` を in-process で使う (高速、デフォルト)。
    Local,
    /// `pokemon-showdown simulate-battle` subprocess を使う (互換性検証用)。
    Showdown,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
pub struct InferencedDataItem {
    pub game_id: usize,
    pub player: Player,
    /// 対応する観測の `request_id` をそのまま返す。game_task でルーティングに使う。
    pub request_id: u64,
    /// softmax 確率 (生)。
    pub policy: [f32; ACTION_DIM],
    /// value head 生出力 (勝率 0..1)。
    pub value: f32,
}

pub(crate) enum ExecutorEnum {
    Observations(PackedBatch),
    Trajectories(Trajectories),
}

/// Pythonスレッドからアクセスされる
pub struct RustAsyncExecutor {
    runtime: Runtime,
    game_senders: Vec<UnboundedSender<InferencedDataItem>>,
    executor_receiver: UnboundedReceiver<ExecutorEnum>,
    prepared_observations: VecDeque<PackedBatch>,
    prepared_trajectories: VecDeque<Trajectories>,
    /// game_id 索引の役割テーブル (0=自己対戦, 非0=敵ゲーム)。敵混合学習で Python が
    /// ブロック境界に `set_roles` で書き換える。各 game_task がバトル開始時に読む。
    roles: Arc<Vec<AtomicU8>>,
}

impl RustAsyncExecutor {
    pub fn new(
        num_games: usize,
        run_seed: u64,
        max_batch_size: usize,
        trajectories_threshold: usize,
        backend: Backend,
        randomize: bool,
        crit_enabled: bool,
        stage: Stage,
        lookahead: LookaheadConfig,
        eval_rule_opponent: bool,
        eval_rule_p1: bool,
    ) -> Self {
        assert!(num_games > 0, "num_games must be greater than 0");
        assert!(
            max_batch_size > 0,
            "max_batch_size must be greater than 0"
        );
        assert!(
            trajectories_threshold > 0,
            "trajectories_threshold must be greater than 0"
        );

        let runtime = Runtime::new().expect("tokio runtime should start");
        let (root_sender, root_receiver) = unbounded_channel();
        let (sender_to_executor, executor_receiver) = unbounded_channel();
        let mut game_senders = Vec::with_capacity(num_games);
        let roles: Arc<Vec<AtomicU8>> =
            Arc::new((0..num_games).map(|_| AtomicU8::new(0)).collect());

        for index in 0..num_games {
            let game_id = index;
            let root_sender = root_sender.clone();
            let (inference_sender, inference_receiver) = unbounded_channel();
            game_senders.push(inference_sender);
            let roles = roles.clone();

            runtime.spawn(async move {
                game_start(
                    game_id,
                    run_seed,
                    backend,
                    randomize,
                    crit_enabled,
                    stage,
                    lookahead,
                    eval_rule_opponent,
                    eval_rule_p1,
                    roles,
                    root_sender,
                    GameReceiver::new(inference_receiver),
                )
                .await;
            });
        }
        drop(root_sender);

        runtime.spawn(
            RootTask {
                root_receiver,
                sender_to_executor,
                trajectories: Vec::new(),
                max_batch_size,
                trajectories_threshold,
                batch_stats: Default::default(),
            }
            .run(),
        );

        Self {
            runtime,
            game_senders,
            executor_receiver,
            prepared_observations: VecDeque::new(),
            prepared_trajectories: VecDeque::new(),
            roles,
        }
    }

    /// 敵混合学習用の役割テーブルを更新する。`roles[game_id]` = 0 で自己対戦、
    /// 非0 で敵ゲーム (P2 を policy-only 化 + trajectory に enemy_game タグ)。
    /// 各 game_task は次バトル開始時に読むので、反映は in-flight バトル完了後。
    pub fn set_roles(&self, roles: &[i64]) -> Result<(), ExecutorError> {
        if roles.len() != self.roles.len() {
            return Err(ExecutorError::InferenceShapeMismatch);
        }
        for (slot, &value) in self.roles.iter().zip(roles.iter()) {
            slot.store(value.clamp(0, u8::MAX as i64) as u8, Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn is_ready(&mut self) -> bool {
        if !self.prepared_observations.is_empty() {
            return true;
        }
        self.drain_executor_receiver();
        !self.prepared_observations.is_empty()
    }

    pub fn trajectories_ready(&mut self) -> bool {
        if !self.prepared_trajectories.is_empty() {
            return true;
        }
        self.drain_executor_receiver();
        !self.prepared_trajectories.is_empty()
    }

    /// 観測バッチをパック済みテンソル形で受け取る。エンコード+パックは
    /// RootTask (tokio ワーカー) 側で完了しており、ここでは取り出すだけ。
    pub fn recv_packed(&mut self) -> Result<PackedBatch, ExecutorError> {
        if self.prepared_observations.is_empty() {
            self.drain_executor_receiver();
        }
        self.prepared_observations
            .pop_front()
            .ok_or(ExecutorError::NoPreparedData)
    }

    /// 推論結果を配列で受け取り、行ごとに game チャネルへ振り分ける。
    /// `policy` は (B, ACTION_DIM) 行優先フラット、`value` は (B,)。
    /// empty 行 (`recv_encoded` の `empty_*` をエコーバックしたもの) は
    /// ダミー policy/value で ack を返す (game_task 側は中身を見ない)。
    #[allow(clippy::too_many_arguments)]
    pub fn send_inference(
        &self,
        game_id: &[i64],
        player: &[i64],
        request_id: &[i64],
        policy: &[f32],
        value: &[f32],
        empty_game_id: &[i64],
        empty_player: &[i64],
        empty_request_id: &[i64],
    ) -> Result<(), ExecutorError> {
        let rows = game_id.len();
        if player.len() != rows
            || request_id.len() != rows
            || value.len() != rows
            || policy.len() != rows * ACTION_DIM
            || empty_player.len() != empty_game_id.len()
            || empty_request_id.len() != empty_game_id.len()
        {
            return Err(ExecutorError::InferenceShapeMismatch);
        }
        for row in 0..rows {
            let mut pi = [0.0f32; ACTION_DIM];
            pi.copy_from_slice(&policy[row * ACTION_DIM..(row + 1) * ACTION_DIM]);
            self.route_item(game_id[row], player[row], request_id[row], pi, value[row])?;
        }
        for row in 0..empty_game_id.len() {
            self.route_item(
                empty_game_id[row],
                empty_player[row],
                empty_request_id[row],
                [0.5; ACTION_DIM],
                0.5,
            )?;
        }
        Ok(())
    }

    fn route_item(
        &self,
        game_id: i64,
        player: i64,
        request_id: i64,
        policy: [f32; ACTION_DIM],
        value: f32,
    ) -> Result<(), ExecutorError> {
        let player = match player {
            0 => Player::P1,
            1 => Player::P2,
            other => return Err(ExecutorError::UnknownPlayerIndex(other)),
        };
        let game_id = usize::try_from(game_id)
            .map_err(|_| ExecutorError::UnknownGameId(usize::MAX))?;
        let item = InferencedDataItem {
            game_id,
            player,
            request_id: request_id as u64,
            policy,
            value,
        };
        let sender = self
            .game_senders
            .get(game_id)
            .ok_or(ExecutorError::UnknownGameId(game_id))?;
        sender.send(item).map_err(|_| ExecutorError::GameClosed)
    }

    pub fn recv_trajectories_json(&mut self) -> Result<String, ExecutorError> {
        if self.prepared_trajectories.is_empty() {
            self.drain_executor_receiver();
        }
        let trajectories = self
            .prepared_trajectories
            .pop_front()
            .ok_or(ExecutorError::NoPreparedTrajectories)?;
        serde_json::to_string(&trajectories).map_err(ExecutorError::Json)
    }

    pub fn game_count(&self) -> usize {
        self.game_senders.len()
    }

    pub fn runtime_handle_count(&self) -> usize {
        self.runtime.metrics().num_workers()
    }

    fn drain_executor_receiver(&mut self) {
        loop {
            match self.executor_receiver.try_recv() {
                Ok(ExecutorEnum::Observations(batch)) => {
                    self.prepared_observations.push_back(batch);
                }
                Ok(ExecutorEnum::Trajectories(trajectories)) => {
                    self.prepared_trajectories.push_back(trajectories);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                    break;
                }
            }
        }
    }
}
