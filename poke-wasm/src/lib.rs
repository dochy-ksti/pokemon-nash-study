//! ブラウザ (WebAssembly) 向けの薄いバトル API。`poke-sho-rust` のターン解決・
//! ダメージ計算をそのまま公開し、乱数・急所ありの本番設定で1ターンずつ進める。
//! AI の着手は別途 JS がポリシーテーブルを引いて決めるため、本 crate は推論を持たない。

mod rng;
mod view;

use poke_sho_rust::battle::{BattleState, Choice, Player};
use poke_sho_rust::damage::{DamageInput, calc_damage};
use poke_sho_rust::scenario::{MoveId, Stage, TeamId};
use poke_sho_rust::turn::{apply_forced_switches, apply_turn};
use rng::WasmRng;
use serde::Serialize;
use view::{StateView, poke_type_name, state_view};
use wasm_bindgen::prelude::*;

fn team_of(t: u8) -> TeamId {
    match t {
        0 => TeamId::Team1,
        _ => TeamId::Team2,
    }
}

fn player_of(p: u8) -> Player {
    match p {
        0 => Player::P1,
        _ => Player::P2,
    }
}

/// `kind` 0=技(arg=`MoveId::index`)/1=交代(arg=パーティ index) を `Choice` へ。
fn choice_of(kind: u8, arg: u8) -> Choice {
    match kind {
        0 => Choice::Move(MoveId::from_index(arg as usize).expect("bad move index")),
        _ => Choice::Switch(arg as usize),
    }
}

#[derive(Serialize)]
struct DamageRange {
    min_hp: i32,
    max_hp: i32,
    min_pct: f32,
    max_pct: f32,
    effectiveness: f64,
}

#[derive(Serialize)]
struct StepResult {
    events: Vec<poke_sho_rust::event::Event>,
    state: StateView,
}

/// 1 バトルのラッパ。`BattleState` は `Copy` なので `step` で丸ごと差し替える。
#[wasm_bindgen]
pub struct Battle {
    state: BattleState,
}

#[wasm_bindgen]
impl Battle {
    /// 満タンで開始する。`stage` は "3b"/"3c"、team は 0=Team1/1=Team2、active は先発 index。
    #[wasm_bindgen(constructor)]
    pub fn new(stage: &str, team1: u8, active1: u8, team2: u8, active2: u8) -> Battle {
        let stage = Stage::from_short_name(stage).expect("unknown stage");
        let state = BattleState::new_with_teams(
            stage,
            (team_of(team1), active1 as usize),
            (team_of(team2), active2 as usize),
        );
        Battle { state }
    }

    /// 任意 HP を設定する (初期局面をカスタムしたい場合)。side 0/1・member はパーティ index。
    #[wasm_bindgen(js_name = setHp)]
    pub fn set_hp(&mut self, side: u8, member: u8, hp: i32) {
        let mon = &mut self.state.parties[side as usize].members[member as usize];
        mon.hp = hp.clamp(0, mon.max_hp);
    }

    /// 現在状態の表示ビュー (JS が描画・テーブル index 計算に使う)。
    #[wasm_bindgen]
    pub fn snapshot(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&state_view(&self.state)).unwrap()
    }

    /// `player` の合法手を `[kind, arg, kind, arg, ...]` の平坦配列で返す。
    #[wasm_bindgen]
    pub fn legal(&self, player: u8) -> Vec<u8> {
        let mut out = Vec::new();
        for c in self.state.legal_choices(player_of(player)) {
            match c {
                Choice::Move(m) => {
                    out.push(0);
                    out.push(m.index() as u8);
                }
                Choice::Switch(i) => {
                    out.push(1);
                    out.push(i as u8);
                }
            }
        }
        out
    }

    /// `attacker` が技スロット `move_slot` を、居座り相手の active に当てた場合の
    /// 通常時(急所なし) min–max ダメージと相性倍率。
    #[wasm_bindgen(js_name = damageRange)]
    pub fn damage_range(&self, attacker: u8, move_slot: u8) -> JsValue {
        let atk_p = player_of(attacker);
        let atk = self.state.pokemon(atk_p);
        let def = self.state.pokemon(atk_p.opponent());
        let mv = MoveId::from_index(move_slot as usize).expect("bad move slot").data();
        let mk = |roll: u8| {
            calc_damage(&DamageInput {
                level: atk.level,
                attacker: &atk.stats,
                attacker_types: atk.types,
                defender: &def.stats,
                defender_types: def.types,
                mv: &mv,
                crit: false,
                roll,
            })
        };
        let min_hp = mk(85);
        let max_hp = mk(100);
        let denom = def.max_hp.max(1) as f32;
        let range = DamageRange {
            min_hp,
            max_hp,
            min_pct: 100.0 * min_hp as f32 / denom,
            max_pct: 100.0 * max_hp as f32 / denom,
            effectiveness: def.types.effectiveness(mv.move_type),
        };
        serde_wasm_bindgen::to_value(&range).unwrap()
    }

    /// 両者の手を渡して1ターン解決する。強制交代は控えが一意なので自動解決する。
    /// `seed` は JS 由来 (Math.random) の乱数種。乱数・急所を有効化して本番設定を再現。
    #[wasm_bindgen]
    pub fn step(
        &mut self,
        c1_kind: u8,
        c1_arg: u8,
        c2_kind: u8,
        c2_arg: u8,
        seed: f64,
    ) -> JsValue {
        let mut rng = WasmRng::from_u64(seed.to_bits(), true, true);
        let first = rng.first_player();
        let result = apply_turn(
            self.state,
            choice_of(c1_kind, c1_arg),
            choice_of(c2_kind, c2_arg),
            first,
            &mut rng,
        );
        self.state = result.state;
        let mut events = result.events;

        // 瀕死強制交代: 3b/3c は控え 1 体なので交代先は一意 → 自動解決。
        if self.state.any_forced_switch() {
            let pick = |p: Player| {
                if self.state.needs_forced_switch(p) {
                    self.state
                        .party(p)
                        .switch_targets()
                        .next()
                        .map(Choice::Switch)
                } else {
                    None
                }
            };
            let fr = apply_forced_switches(self.state, pick(Player::P1), pick(Player::P2));
            self.state = fr.state;
            events.extend(fr.events);
        }

        let out = StepResult {
            events,
            state: state_view(&self.state),
        };
        serde_wasm_bindgen::to_value(&out).unwrap()
    }
}

/// PokeType 名の一覧 (JS 側の相性色分け等に使える補助)。未使用でも export しておく。
#[wasm_bindgen(js_name = typeName)]
pub fn type_name(idx: u8) -> String {
    use poke_sho_rust::types::PokeType::*;
    let all = [
        Normal, Fighting, Flying, Poison, Ground, Rock, Bug, Ghost, Steel, Fire, Water,
        Grass, Electric, Psychic, Ice, Dragon, Dark, Fairy,
    ];
    all.get(idx as usize).map(|t| poke_type_name(*t).to_string()).unwrap_or_default()
}
