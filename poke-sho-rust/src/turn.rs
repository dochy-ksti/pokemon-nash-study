//! ターン解決ロジック。通常ターン (`apply_turn`)、瀕死後の強制交代
//! (`apply_forced_switches`)、単一技の決定論的解決 (`execute_action`) を持つ。
//!
//! 状態とパーティの型は `battle` / `party` にあり、ここはそれらを使って局面を
//! 進める純粋ロジックに徹する。交代は全ての技より先に解決し、技は `first_player`
//! の速度順で解決する。瀕死で控えが残る側へは `forced_switch` を立て、呼び出し側が
//! 別ステップで強制交代を解決する (Showdown の force-switch リクエストに対応)。

use crate::battle::{BattleState, Choice, Player};
use crate::battle_rng::BattleRng;
use crate::damage::{DamageInput, calc_damage};
use crate::event::{Event, PokemonRef};
use crate::scenario::MoveId;

/// Result of applying one turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnResult {
    pub state: BattleState,
    pub events: Vec<Event>,
}

fn pokemon_ref(state: &BattleState, player: Player) -> PokemonRef {
    PokemonRef::new(player.index() as u8 + 1, state.pokemon(player).display_name)
}

fn player_name(player: Player) -> &'static str {
    match player {
        Player::P1 => "P1",
        Player::P2 => "P2",
    }
}

fn choice_of(player: Player, p1: Choice, p2: Choice) -> Choice {
    match player {
        Player::P1 => p1,
        Player::P2 => p2,
    }
}

fn apply_damage(state: &mut BattleState, target: Player, amount: i32) -> i32 {
    let mon = state.parties[target.index()].active_mon_mut();
    let hp_before = mon.hp;
    let actual = amount.min(hp_before);
    mon.hp = hp_before - actual;
    actual
}

/// 交代を解決し `Switch` イベントを返す。合法手マスクで弾かれる前提。
fn resolve_switch(state: &mut BattleState, player: Player, idx: usize) -> Vec<Event> {
    let party = &mut state.parties[player.index()];
    assert!(
        idx < party.len,
        "resolve_switch: idx={idx} is out of range (len={})", party.len
    );
    assert!(
        idx != party.active,
        "resolve_switch: idx={idx} is already active"
    );
    assert!(
        party.members[idx].hp > 0,
        "resolve_switch: idx={idx} is fainted"
    );
    party.active = idx;
    let mon = *party.active_mon();
    vec![Event::Switch {
        who: PokemonRef::new(player.index() as u8 + 1, mon.display_name),
        species: mon.name.to_string(),
        hp: mon.hp.max(0) as u32,
        max_hp: mon.max_hp as u32,
    }]
}

/// 1 つの技を決定論的に解決する。`crit` と `roll` を直接渡すので、parity 検証で
/// Showdown の観測値を replay する側でも使える。
pub fn execute_action(
    state: &mut BattleState,
    attacker: Player,
    move_id: MoveId,
    crit: bool,
    roll: u8,
) -> Vec<Event> {
    let mut events = Vec::new();
    let target = attacker.opponent();
    events.push(Event::Move {
        user: pokemon_ref(state, attacker),
        move_id: move_id.data().id.to_string(),
        target: pokemon_ref(state, target),
    });

    let atk = state.pokemon(attacker);
    let def = state.pokemon(target);
    let mv = move_id.data();

    let damage = calc_damage(&DamageInput {
        level: atk.level,
        attacker: &atk.stats,
        attacker_types: atk.types,
        defender: &def.stats,
        defender_types: def.types,
        mv: &mv,
        crit,
        roll,
    });

    if crit && damage > 0 {
        events.push(Event::Crit {
            target: pokemon_ref(state, target),
        });
    }
    apply_damage(state, target, damage);
    let fainted = state.is_fainted(target);
    let (hp, max_hp) = if fainted {
        (0, 0)
    } else {
        (
            state.hp(target).max(0) as u32,
            state.pokemon(target).max_hp as u32,
        )
    };
    events.push(Event::Damage {
        target: pokemon_ref(state, target),
        hp,
        max_hp,
        fainted,
    });
    if fainted {
        events.push(Event::Faint {
            target: pokemon_ref(state, target),
        });
        // パーティ全滅ではじめて勝敗が決まる。
        if state.is_lost(target) {
            events.push(Event::Win {
                player: player_name(attacker).to_string(),
            });
        }
    }
    events
}

fn execute_action_rng<R: BattleRng>(
    state: &mut BattleState,
    attacker: Player,
    move_id: MoveId,
    rng: &mut R,
) -> Vec<Event> {
    let crit = rng.is_crit(0);
    let roll = rng.damage_roll();
    execute_action(state, attacker, move_id, crit, roll)
}

/// 通常ターンを解決する。交代フェーズ (両者・技より先) のあと、技フェーズを
/// `first_player` の速度順で解決する。ターン終了時に場が瀕死かつ控えが残る側へは
/// `forced_switch` フラグを立て、呼び出し側が `apply_forced_switches` で別途解決する。
pub fn apply_turn<R: BattleRng>(
    state: BattleState,
    p1_choice: Choice,
    p2_choice: Choice,
    first_player: Player,
    rng: &mut R,
) -> TurnResult {
    if state.is_done() {
        return TurnResult {
            state,
            events: Vec::new(),
        };
    }

    let mut new_state = state;
    let mut events = Vec::new();
    let order = [first_player, first_player.opponent()];

    // フェーズ 1: 交代 (両者)。交代同士は相互作用がないので順序非依存。
    for &p in &order {
        if let Choice::Switch(idx) = choice_of(p, p1_choice, p2_choice) {
            events.extend(resolve_switch(&mut new_state, p, idx));
        }
    }

    // フェーズ 2: 技 (速度順)。交代した側・瀕死の側は撃たない。
    for &p in &order {
        if new_state.is_done() {
            break;
        }
        if let Choice::Move(move_id) = choice_of(p, p1_choice, p2_choice)
            && !new_state.is_fainted(p)
        {
            events.extend(execute_action_rng(&mut new_state, p, move_id, rng));
        }
    }

    new_state.turn += 1;

    // ターン上限に達したら無条件引き分けで終局する (勝敗未決のまま打ち切り)。
    if new_state.is_draw() {
        events.push(Event::Tie);
        return TurnResult {
            state: new_state,
            events,
        };
    }

    // ターン終了後、瀕死で交代可能な側に強制交代を要求する。
    for p in [Player::P1, Player::P2] {
        new_state.forced_switch[p.index()] =
            new_state.is_fainted(p) && new_state.party(p).has_living_bench();
    }

    TurnResult {
        state: new_state,
        events,
    }
}

/// 瀕死後の強制交代を解決する。`forced_switch` が立っている側にのみ `Choice::Switch`
/// を渡す (立っていない側の選択は無視)。解決後、該当フラグをクリアする。
pub fn apply_forced_switches(
    state: BattleState,
    p1_choice: Option<Choice>,
    p2_choice: Option<Choice>,
) -> TurnResult {
    let mut new_state = state;
    let mut events = Vec::new();
    for (p, choice) in [(Player::P1, p1_choice), (Player::P2, p2_choice)] {
        if new_state.forced_switch[p.index()]
            && let Some(Choice::Switch(idx)) = choice
        {
            let switched = resolve_switch(&mut new_state, p, idx);
            if !switched.is_empty() {
                events.extend(switched);
                new_state.forced_switch[p.index()] = false;
            }
        }
    }
    TurnResult {
        state: new_state,
        events,
    }
}
