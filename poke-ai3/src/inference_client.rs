//! lookahead から policy/value 推論を batch 経由で取得するためのクライアント。
//!
//! 1 ゲーム = 1 推論ルータ。`InferenceClient` は観測を `request_id` 付きで root に送り、
//! 対応する応答 (`InferencedDataItem`) を待つ oneshot を `pending` に登録する。
//! ルータ (game_task の run_one_game) が inference 応答を受け取り、`request_id` で
//! 対応する oneshot へ届ける。複数の rollout が同時に推論待ちできるようになる。

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use poke_env_rust::observation::{Player, StateForPlayer};
use poke_env_rust::oracle::{OracleOut, PolicyOracle};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;

use crate::async_executor::{InferencedDataItem, ACTION_DIM};
use crate::root_task::{PlayerObservation, RootEnumFromGame};

pub(crate) type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<InferencedDataItem>>>>;

#[derive(Clone)]
pub(crate) struct InferenceClient {
    game_id: usize,
    root_sender: UnboundedSender<RootEnumFromGame>,
    pending: PendingMap,
    counter: Arc<AtomicU64>,
}

impl InferenceClient {
    pub(crate) fn new(
        game_id: usize,
        root_sender: UnboundedSender<RootEnumFromGame>,
        pending: PendingMap,
    ) -> Self {
        Self {
            game_id,
            root_sender,
            pending,
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Showdown バックエンド用: lookahead を介さず 1 回だけ推論する。
    pub(crate) async fn infer_public(
        &self,
        player: Player,
        state: StateForPlayer,
    ) -> InferencedDataItem {
        self.request(player, state).await
    }

    /// 観測を送って応答を待つ。チャネルが閉じたら一様分布で fallback する。
	/// この関数は PolicyOracle trait越しに呼び出される
    async fn request(&self, player: Player, state: StateForPlayer) -> InferencedDataItem {
        self.round_trip(player, Some(state)).await
    }

    /// empty (穴埋め用ダミー) を送って ack を待つ。応答内容は使わない。
    /// state=None で送るため Python は GPU forward に載せず request_id をエコーバックする。
    async fn empty_round_trip(&self, player: Player) {
        let _ = self.round_trip(player, None).await;
    }

    /// 観測 (実 or empty) を送って応答 (実推論 or ack) を待つ共通処理。
    async fn round_trip(
        &self,
        player: Player,
        state: Option<StateForPlayer>,
    ) -> InferencedDataItem {
        let request_id = self.counter.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        {
			// AIが書いた Arc<Mutex<HashMap>>を使うこれは俺は嫌だが、実害はないのでよいものとする。
            let mut pending = self.pending.lock().expect("pending map poisoned");
            pending.insert(request_id, tx);
        }
        let observation = PlayerObservation {
            game_id: self.game_id,
            player,
            request_id,
            state,
        };
        if self
            .root_sender
            .send(RootEnumFromGame::Data(observation))
            .is_err()
        {
            self.pending.lock().expect("pending map poisoned").remove(&request_id);
            return fallback_item(self.game_id, player, request_id);
        }
        match rx.await {
            Ok(item) => item,
            Err(_) => fallback_item(self.game_id, player, request_id),
        }
    }
}

fn fallback_item(game_id: usize, player: Player, request_id: u64) -> InferencedDataItem {
    InferencedDataItem {
        game_id,
        player,
        request_id,
        policy: [1.0 / ACTION_DIM as f32; ACTION_DIM],
        value: 0.5,
    }
}

impl PolicyOracle for InferenceClient {
    async fn infer(&self, state: StateForPlayer, player: Player) -> OracleOut {
        let item = self.request(player, state).await;
        OracleOut {
            policy: item.policy,
            value: item.value,
        }
    }

    async fn ack_empty(&self, player: Player) {
        self.empty_round_trip(player).await;
    }
}
