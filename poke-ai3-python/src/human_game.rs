//! 人間 vs AI 対戦 GUI 用の単一対戦バインディング (同期駆動)。
//!
//! 学習用の `RustAsyncExecutor` (tokio + バッチ推論ルーティング) とは分離した、
//! 1 試合をインタラクティブに進めるための最小 API。状態は Rust 側 (`BattleState`)
//! が真実として保持し、Python へは player 視点の観測 (`StateForPlayer`) だけを切り
//! 出して渡す (相手の控え・技の隠匿境界を自然に守る)。
//!
//! 条件は学習・評価と揃える: ダメージは `MaxRoll` (最大ロール・急所なし) で決定論。
//! 行動順コイン (`first_player`) のみ呼び出し側 (Python) が毎ターン公平に決める。
//! Stage3b は必ずクロスチーム (Team1 vs Team2) で組む。P1=人間, P2=AI。

use poke_env_rust::observation::{
    BattleState, Choice, Player, Stage, TeamId, action_to_choice, observation_for,
};
use poke_sho_rust::battle::{apply_forced_switches, apply_turn};
use poke_sho_rust::battle_rng::MaxRoll;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn player_of(num: u8) -> PyResult<Player> {
    match num {
        1 => Ok(Player::P1),
        2 => Ok(Player::P2),
        other => Err(PyValueError::new_err(format!(
            "player must be 1 (human) or 2 (AI), got {other}"
        ))),
    }
}

fn parse_team(name: &str) -> PyResult<TeamId> {
    match name {
        "team1" => Ok(TeamId::Team1),
        "team2" => Ok(TeamId::Team2),
        other => Err(PyValueError::new_err(format!(
            "team must be 'team1' or 'team2', got '{other}'"
        ))),
    }
}

fn parse_stage(name: &str) -> PyResult<Stage> {
    Stage::from_short_name(name)
        .filter(|s| s.is_party())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "stage must be a party stage ('3b' or '3c'), got '{name}'"
            ))
        })
}

/// 強制交代待ちの側について、唯一の合法交代手を返す (3b/3c は控え 1 体なので一択)。
fn forced_choice(state: &BattleState, player: Player) -> Option<Choice> {
    if state.needs_forced_switch(player) {
        state.legal_choices(player).first().copied()
    } else {
        None
    }
}

fn winner_num(state: &BattleState) -> Option<u8> {
    state.winner().map(|p| p.index() as u8 + 1)
}

/// 1 試合分のインタラクティブセッション。P1=人間, P2=AI。
#[pyclass(name = "HumanGame")]
pub struct PyHumanGame {
    state: BattleState,
}

#[pymethods]
impl PyHumanGame {
    /// 新規対戦を開始する。`human_team` は人間 (P1) のチーム ("team1"/"team2")、
    /// AI (P2) は必ず逆チーム。`human_lead`/`ai_lead` は先発の**宣言順**種族 index
    /// (Cloyster=0, Goodra=1)。`stage` は "3b" または "3c" (デフォルト "3b")。
    #[new]
    #[pyo3(signature = (human_team, human_lead, ai_lead, stage = "3b"))]
    fn new(human_team: &str, human_lead: usize, ai_lead: usize, stage: &str) -> PyResult<Self> {
        let stage = parse_stage(stage)?;
        let human = parse_team(human_team)?;
        let ai = match human {
            TeamId::Team1 => TeamId::Team2,
            TeamId::Team2 => TeamId::Team1,
        };
        if human_lead > 1 || ai_lead > 1 {
            return Err(PyValueError::new_err(
                "lead index must be 0 (Cloyster) or 1 (Goodra)",
            ));
        }
        let state = BattleState::new_with_teams(stage, (human, human_lead), (ai, ai_lead));
        Ok(Self { state })
    }

    /// 指定 player 視点の観測 (`StateForPlayer`) を JSON で返す。
    /// player=1 が人間 UI 用、player=2 が AI 推論用。
    fn observation(&self, player: u8) -> PyResult<String> {
        let p = player_of(player)?;
        serde_json::to_string(&observation_for(&self.state, p))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// 終局しているか。
    fn done(&self) -> bool {
        self.state.is_done()
    }

    /// 勝者 (1=人間, 2=AI, None=未決 or 引き分け)。
    fn winner(&self) -> Option<u8> {
        winner_num(&self.state)
    }

    /// ターン上限による引き分けで終局したか。
    fn is_draw(&self) -> bool {
        self.state.is_draw()
    }

    /// 現在ターン番号。
    fn turn(&self) -> u32 {
        self.state.turn
    }

    /// 1 ターン進める。`human_action`/`ai_action` は ACTION_DIM index、
    /// `human_first` は行動順コイン (true=人間先手)。瀕死後の強制交代 (一択) は
    /// 自動解決し、発生イベント列を含めて返す。
    ///
    /// 返値は JSON: `{ "events": [...], "done": bool, "winner": 1|2|null, "draw": bool }`。
    fn step(&mut self, human_action: usize, ai_action: usize, human_first: bool) -> PyResult<String> {
        if self.state.is_done() {
            return Err(PyValueError::new_err("game is already over"));
        }
        let c1 = action_to_choice(self.state.party(Player::P1), human_action);
        let c2 = action_to_choice(self.state.party(Player::P2), ai_action);
        let first = if human_first { Player::P1 } else { Player::P2 };

        let mut rng = MaxRoll;
        let res = apply_turn(self.state, c1, c2, first, &mut rng);
        let mut events = res.events;
        let mut state = res.state;

        // 強制交代 (瀕死後) は一択なので自動解決する。
        if state.any_forced_switch() {
            let fc1 = forced_choice(&state, Player::P1);
            let fc2 = forced_choice(&state, Player::P2);
            let forced = apply_forced_switches(state, fc1, fc2);
            events.extend(forced.events);
            state = forced.state;
        }
        self.state = state;

        let out = serde_json::json!({
            "events": events,
            "done": self.state.is_done(),
            "winner": winner_num(&self.state),
            "draw": self.state.is_draw(),
            "first": if human_first { 1 } else { 2 },
        });
        serde_json::to_string(&out).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
