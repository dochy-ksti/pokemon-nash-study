from typing import Any

NUM_MOVES: int
MAX_MOVE_SLOTS: int
NUM_BENCH: int
ACTION_DIM: int
SPECIES_VOCAB: int
MOVE_VOCAB: int
SPECIES_VOCAB_CAP: int
MOVE_VOCAB_CAP: int
TYPE_VOCAB: int
SPECIES_TABLE_TSV: str
MOVE_TABLE_TSV: str
TYPE_TABLE_TSV: str

class RustAsyncExecutor:
    def __init__(
        self,
        num_games: int,
        max_batch_size: int | None = ...,
        trajectories_threshold: int | None = ...,
        backend: str = ...,
        randomize: bool = ...,
        crit_enabled: bool = ...,
        stage: str = ...,
        sims: int = ...,
        sim_concurrency: int = ...,
        search_turn_min: int = ...,
        search_turn_max: int = ...,
        eval_rule_opponent: bool = ...,
        eval_rule_p1: bool = ...,
        battle_seed: int = ...,
    ) -> None: ...
    def is_ready(self) -> bool: ...
    def trajectories_ready(self) -> bool: ...
    def recv_observations(self) -> dict[str, Any]: ...
    def recv_observations_with_json(self) -> tuple[dict[str, Any], str]: ...
    def send_inference(
        self,
        game_id: Any,
        player: Any,
        request_id: Any,
        policy: Any,
        value: Any,
        empty_game_id: Any,
        empty_player: Any,
        empty_request_id: Any,
    ) -> None: ...
    def recv_trajectories(self) -> str: ...

class HumanGame(Any): ...

def get_rust_async_executor_wrapper(
    num_games: int,
    max_batch_size: int | None = ...,
    trajectories_threshold: int | None = ...,
    backend: str = ...,
    randomize: bool = ...,
    crit_enabled: bool = ...,
    stage: str = ...,
    sims: int = ...,
    sim_concurrency: int = ...,
    search_turn_min: int = ...,
    search_turn_max: int = ...,
    eval_rule_opponent: bool = ...,
    eval_rule_p1: bool = ...,
    battle_seed: int = ...,
) -> RustAsyncExecutor: ...
