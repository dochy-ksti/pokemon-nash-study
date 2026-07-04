#[derive(Debug)]
pub enum ExecutorError {
    Json(serde_json::Error),
    GameClosed,
    NoPreparedData,
    NoPreparedTrajectories,
    UnknownGameId(usize),
    UnknownPlayerIndex(i64),
    InferenceShapeMismatch,
}

impl From<serde_json::Error> for ExecutorError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl std::fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutorError::Json(error) => write!(f, "{error}"),
            ExecutorError::GameClosed => write!(f, "game receiver is closed"),
            ExecutorError::NoPreparedData => write!(f, "no prepared observation batch is ready"),
            ExecutorError::NoPreparedTrajectories => {
                write!(f, "no prepared trajectories are ready")
            }
            ExecutorError::UnknownGameId(game_id) => {
                write!(f, "unknown game_id {}", game_id)
            }
            ExecutorError::UnknownPlayerIndex(index) => {
                write!(f, "unknown player index {} (expected 0 or 1)", index)
            }
            ExecutorError::InferenceShapeMismatch => {
                write!(f, "inference array lengths are inconsistent")
            }
        }
    }
}

impl std::error::Error for ExecutorError {}
