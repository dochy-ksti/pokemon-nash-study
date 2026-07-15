//! 静的Web対戦アプリ用ポリシーテーブルの状態列挙器。
//!
//! party stage は「両サイド2体(Cloyster/Goodra)の各HP + どちらがアクティブ + チーム構成」だけで
//! 状態が決まる。HP を `hp_buckets` 段に離散化し、正準列挙順(= 密 index)で全状態を回して
//! AI(=P1 視点)の観測 `StateForPlayer` を作る。Python ドライバがこれを infer して
//! P(交代) を密配列へ焼く。runtime(JS)は同じ index 式でテーブルを引く。
//!
//! クロスチーム限定(学習分布と一致)なので `opp_team = 1 - ai_team` と一意に決まり、
//! opp_team 次元は持たない。正準列挙順(最上位→最下位): ai_team[2], ai_active[2],
//! ai_hp_cloyster[H], ai_hp_goodra[H], opp_active[2], opp_hp_cloyster[H], opp_hp_goodra[H]。
//! 総数 = 8 * H^4。radix はこの並び。

use crate::encoded::encoded_batch_to_dict;
use poke_ai3::obs_encode::encode_states;
use poke_env_rust::observation::{StateForPlayer, observation_for};
use poke_sho_rust::battle::{BattleState, Player};
use poke_sho_rust::scenario::{Stage, TeamId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn team_of(v: u64) -> TeamId {
    if v == 0 { TeamId::Team1 } else { TeamId::Team2 }
}

/// バケット index → 代表 HP。k=0→0(瀕死), k=H-1→満タン。runtime の逆写像と一致させる。
fn bucket_to_hp(k: u64, max_hp: i32, buckets: u64) -> i32 {
    if buckets <= 1 {
        return max_hp;
    }
    ((k as f64) * (max_hp as f64) / ((buckets - 1) as f64)).round() as i32
}

/// dense index 範囲 `[start, start+count)` を列挙し、有効状態(両アクティブが生存)のみ
/// エンコードした numpy dict と、その dense index 列を返す。無効(アクティブ瀕死)は除外。
#[pyfunction]
pub fn enumerate_policy_batch<'py>(
    py: Python<'py>,
    stage_str: &str,
    hp_buckets: u64,
    start: u64,
    count: u64,
) -> PyResult<(Bound<'py, PyDict>, Vec<u32>)> {
    let stage = Stage::from_short_name(stage_str)
        .ok_or_else(|| PyValueError::new_err(format!("unknown stage '{stage_str}'")))?;
    if !stage.is_party() {
        return Err(PyValueError::new_err("policy table needs a party stage"));
    }
    let h = hp_buckets;
    let total = 8 * h * h * h * h;
    let end = (start + count).min(total);

    let mut states: Vec<StateForPlayer> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for k in start..end {
        // 密 index を分解 (最下位から順に取り出す)。並びは canonical order の逆順。
        let mut r = k;
        let opp_hp_g = r % h; r /= h;
        let opp_hp_c = r % h; r /= h;
        let opp_active = r % 2; r /= 2;
        let ai_hp_g = r % h; r /= h;
        let ai_hp_c = r % h; r /= h;
        let ai_active = r % 2; r /= 2;
        let ai_team = r % 2;
        // クロスチーム限定: 相手は必ず反対の技構成。
        let opp_team = 1 - ai_team;

        // アクティブが瀕死(bucket 0)になる組合せは実局面に現れない → 除外。
        let ai_active_bucket = if ai_active == 0 { ai_hp_c } else { ai_hp_g };
        let opp_active_bucket = if opp_active == 0 { opp_hp_c } else { opp_hp_g };
        if ai_active_bucket == 0 || opp_active_bucket == 0 {
            continue;
        }

        // AI = P1, 相手 = P2 で構築(満タン)→ 各メンバー HP をバケット代表値へ。
        let mut st = BattleState::new_with_teams(
            stage,
            (team_of(ai_team), ai_active as usize),
            (team_of(opp_team), opp_active as usize),
        );
        set_member_hp(&mut st, 0, 0, ai_hp_c, h);
        set_member_hp(&mut st, 0, 1, ai_hp_g, h);
        set_member_hp(&mut st, 1, 0, opp_hp_c, h);
        set_member_hp(&mut st, 1, 1, opp_hp_g, h);

        states.push(observation_for(&st, Player::P1));
        indices.push(k as u32);
    }

    let dict = encoded_batch_to_dict(py, encode_states(&states))?;
    Ok((dict, indices))
}

fn set_member_hp(st: &mut BattleState, side: usize, member: usize, bucket: u64, buckets: u64) {
    let mon = &mut st.parties[side].members[member];
    mon.hp = bucket_to_hp(bucket, mon.max_hp, buckets);
}
