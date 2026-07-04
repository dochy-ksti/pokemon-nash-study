//! Pokemon Showdown プロトコル文字列を共通 [`Event`] 列・型付き [`Request`] へ
//! 変換するパーサ。
//!
//! - `update` ブロックの各行を認識できる範囲で [`Event`] にし、語彙外の行
//!   (`|t:|` `|upkeep|` `|-supereffective|` `|gen|` `|player|` 等) は破棄する。
//!   `|switch|`/`|drag|`/`|turn|` は認識する (交代・ターン境界)。
//! - `|split|pN` は「直後の secret 行(実数 HP)を採用し、続く public 行を破棄」する。
//!   omniscient ストリームは secret 行に実数 HP を含むため、ここからアンカー用の
//!   HP を取得できる。
//! - `|request|{json}` は選択可能な行動だけを抜き出して [`Request`] にする。

use poke_sho_rust::event::{Event, PokemonRef};
use serde_json::Value;

use crate::showdown_trait::{MoveOption, Request, TeamMember};

/// Showdown の `toID`: 小文字化し英数字以外を除去する。
pub fn to_id(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// ident を [`PokemonRef`] に分解する。active-field 形式 `p1a: Mew`(スロット
/// 文字あり)と request/party 形式 `p1: Mew`(スロット文字なし)の両方を受け付け、
/// スロット文字は捨てて player 番号と name のみを取り出す。
pub fn parse_ident(s: &str) -> Option<PokemonRef> {
    let (pos, name) = s.split_once(": ")?;
    let bytes = pos.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'p' {
        return None;
    }
    let player = (bytes[1] as char).to_digit(10)? as u8;
    // bytes[2..] があればスロット文字だが singles では無視する。
    Some(PokemonRef::new(player, name.to_string()))
}

/// `138/175` や `0 fnt`、`138/175 brn` などの condition を
/// `(hp, max_hp, fainted)` に分解する。
fn parse_condition(cond: &str) -> Option<(u32, u32, bool)> {
    let hp_part = cond.split_whitespace().next().unwrap_or(cond);
    let fainted = cond.contains("fnt");
    if let Some((cur, max)) = hp_part.split_once('/') {
        let cur: u32 = cur.parse().ok()?;
        let max: u32 = max.parse().ok()?;
        Some((cur, max, fainted || cur == 0))
    } else {
        // `0 fnt` の `hp_part` は `0`。max は不明なので 0 を返す。
        let cur: u32 = hp_part.parse().ok()?;
        Some((cur, 0, fainted || cur == 0))
    }
}

/// 1 行を共通 [`Event`] に変換する。認識できない行は `None`(破棄)。
pub fn parse_event_line(line: &str) -> Option<Event> {
    let trimmed = line.strip_prefix('|')?;
    let mut parts = trimmed.split('|');
    let kind = parts.next()?;
    match kind {
        "move" => {
            let user = parse_ident(parts.next()?)?;
            let move_id = to_id(parts.next()?);
            let target = parse_ident(parts.next()?)?;
            Some(Event::Move {
                user,
                move_id,
                target,
            })
        }
        "-crit" => Some(Event::Crit {
            target: parse_ident(parts.next()?)?,
        }),
        "-damage" => {
            let target = parse_ident(parts.next()?)?;
            let (hp, max_hp, fainted) = parse_condition(parts.next()?)?;
            Some(Event::Damage {
                target,
                hp,
                max_hp,
                fainted,
            })
        }
        "faint" => Some(Event::Faint {
            target: parse_ident(parts.next()?)?,
        }),
        // `|switch|<who>|<species>, L<lvl>|<cur>/<max>`。DETAILS 先頭が forme 含む
        // 正式種族名 (例 "Goodra-Hisui")。自発交代・強制交代・先発リードで共通。
        "switch" | "drag" => {
            let who = parse_ident(parts.next()?)?;
            let details = parts.next()?;
            let species = details.split(',').next().unwrap_or(details).trim().to_string();
            let (hp, max_hp, _fainted) = parse_condition(parts.next()?)?;
            Some(Event::Switch {
                who,
                species,
                hp,
                max_hp,
            })
        }
        // `|turn|<n>`。ターン境界マーカー (分割用の区切り)。
        "turn" => Some(Event::Turn {
            n: parts.next()?.trim().parse().ok()?,
        }),
        "win" => Some(Event::Win {
            player: parts.next()?.to_string(),
        }),
        "tie" => Some(Event::Tie),
        _ => None,
    }
}

/// `update` ブロックの行列を共通 [`Event`] 列へ。`|split|pN` は直後の secret 行を
/// 採用し public 行を捨てる。
pub fn parse_update_block(buf: &[String]) -> Vec<Event> {
    let mut events = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        if buf[i].starts_with("|split|") {
            // 直後が secret 行(実数 HP)、その次が public 行(破棄)。
            if let Some(secret) = buf.get(i + 1) {
                if let Some(ev) = parse_event_line(secret) {
                    events.push(ev);
                }
            }
            i += 3;
        } else {
            if let Some(ev) = parse_event_line(&buf[i]) {
                events.push(ev);
            }
            i += 1;
        }
    }
    events
}

/// `|request|{json}` の payload を型付き [`Request`] に変換する。
pub fn parse_request_json(json: &str) -> Request {
    let Ok(v) = serde_json::from_str::<Value>(json) else {
        return Request::Wait;
    };
    if v.get("teamPreview").and_then(Value::as_bool).unwrap_or(false) {
        return Request::TeamPreview;
    }
    if v.get("wait").and_then(Value::as_bool).unwrap_or(false) {
        return Request::Wait;
    }
    // forceSwitch: 瀕死後の強制交代。交代手だけが合法で、控えの状態を `team` に持つ。
    if v.get("forceSwitch").is_some() {
        return Request::ForceSwitch {
            team: parse_team(&v),
        };
    }
    let moves = v
        .get("active")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(|active| active.get("moves"))
        .and_then(Value::as_array)
        .map(|moves| {
            moves
                .iter()
                .map(|m| MoveOption {
                    id: m
                        .get("id")
                        .and_then(Value::as_str)
                        .map(to_id)
                        .unwrap_or_default(),
                    disabled: m.get("disabled").and_then(Value::as_bool).unwrap_or(false),
                })
                .collect()
        });
    match moves {
        Some(moves) => Request::Move {
            moves,
            team: parse_team(&v),
        },
        None => Request::Wait,
    }
}

/// `side.pokemon[]` を自軍パーティの [`TeamMember`] 列へ。各要素の `ident`
/// (`p1: Mew`)と `condition`(`139/175` / `0 fnt`)を分解する。認識できない要素は
/// 飛ばす。
fn parse_team(v: &Value) -> Vec<TeamMember> {
    v.get("side")
        .and_then(|side| side.get("pokemon"))
        .and_then(Value::as_array)
        .map(|mons| {
            mons.iter()
                .filter_map(|m| {
                    let mon = parse_ident(m.get("ident").and_then(Value::as_str)?)?;
                    let (hp, max_hp, fainted) =
                        parse_condition(m.get("condition").and_then(Value::as_str)?)?;
                    Some(TeamMember {
                        mon,
                        hp,
                        max_hp,
                        fainted,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_id_normalizes() {
        assert_eq!(to_id("Hyper Beam"), "hyperbeam");
        assert_eq!(to_id("Tackle"), "tackle");
        assert_eq!(to_id("U-turn"), "uturn");
    }

    #[test]
    fn parse_ident_splits_position_and_name() {
        // active-field 形式(スロット文字あり)。
        assert_eq!(parse_ident("p1a: Mew"), Some(PokemonRef::new(1, "Mew")));
        assert_eq!(parse_ident("p2b: Nick Name"), Some(PokemonRef::new(2, "Nick Name")));
        // request/party 形式(スロット文字なし)も同じ PokemonRef になる。
        assert_eq!(parse_ident("p1: Mew"), Some(PokemonRef::new(1, "Mew")));
        assert_eq!(parse_ident("garbage"), None);
    }

    #[test]
    fn parse_move_and_damage_lines() {
        assert_eq!(
            parse_event_line("|move|p1a: Mew|Tackle|p2a: Mew"),
            Some(Event::Move {
                user: PokemonRef::new(1, "Mew"),
                move_id: "tackle".to_string(),
                target: PokemonRef::new(2, "Mew"),
            })
        );
        assert_eq!(
            parse_event_line("|-damage|p2a: Mew|138/175"),
            Some(Event::Damage {
                target: PokemonRef::new(2, "Mew"),
                hp: 138,
                max_hp: 175,
                fainted: false,
            })
        );
        assert_eq!(parse_event_line("|-supereffective|p2a: Mew"), None);
        assert_eq!(parse_event_line("|turn|3"), Some(Event::Turn { n: 3 }));
    }

    #[test]
    fn parse_switch_line_keeps_forme_species() {
        // DETAILS 先頭が forme 含む正式名。Hisui を厳密に保持する。
        assert_eq!(
            parse_event_line("|switch|p1a: Goodra|Goodra-Hisui, L50|175/175"),
            Some(Event::Switch {
                who: PokemonRef::new(1, "Goodra"),
                species: "Goodra-Hisui".to_string(),
                hp: 175,
                max_hp: 175,
            })
        );
        // 原種 Goodra は別 species 文字列として残る (吸収しない)。
        assert_eq!(
            parse_event_line("|switch|p2a: Goodra|Goodra, L50|150/150"),
            Some(Event::Switch {
                who: PokemonRef::new(2, "Goodra"),
                species: "Goodra".to_string(),
                hp: 150,
                max_hp: 150,
            })
        );
    }

    #[test]
    fn split_block_takes_secret_line() {
        let buf = vec![
            "|move|p1a: Mew|Strength|p2a: Mew".to_string(),
            "|split|p2".to_string(),
            "|-damage|p2a: Mew|138/175".to_string(), // secret (exact)
            "|-damage|p2a: Mew|81/100".to_string(),  // public (percent), dropped
        ];
        let events = parse_update_block(&buf);
        assert_eq!(
            events,
            vec![
                Event::Move {
                    user: PokemonRef::new(1, "Mew"),
                    move_id: "strength".to_string(),
                    target: PokemonRef::new(2, "Mew"),
                },
                Event::Damage {
                    target: PokemonRef::new(2, "Mew"),
                    hp: 138,
                    max_hp: 175,
                    fainted: false,
                },
            ]
        );
    }

    #[test]
    fn request_kinds() {
        assert_eq!(
            parse_request_json(r#"{"teamPreview":true,"side":{}}"#),
            Request::TeamPreview
        );
        assert_eq!(parse_request_json(r#"{"wait":true}"#), Request::Wait);
        assert_eq!(
            parse_request_json(
                r#"{"forceSwitch":[true],"side":{"pokemon":[{"ident":"p1: Mew","condition":"0 fnt"},{"ident":"p1: Ditto","condition":"175/175"}]}}"#
            ),
            Request::ForceSwitch {
                team: vec![
                    TeamMember { mon: PokemonRef::new(1, "Mew"), hp: 0, max_hp: 0, fainted: true },
                    TeamMember { mon: PokemonRef::new(1, "Ditto"), hp: 175, max_hp: 175, fainted: false },
                ],
            }
        );
        assert_eq!(
            parse_request_json(
                r#"{"active":[{"moves":[{"id":"tackle","disabled":false},{"id":"strength","disabled":false}]}],"side":{"pokemon":[{"ident":"p1: Mew","condition":"139/175"}]}}"#
            ),
            Request::Move {
                moves: vec![
                    MoveOption { id: "tackle".to_string(), disabled: false },
                    MoveOption { id: "strength".to_string(), disabled: false },
                ],
                team: vec![TeamMember {
                    mon: PokemonRef::new(1, "Mew"),
                    hp: 139,
                    max_hp: 175,
                    fainted: false,
                }],
            }
        );
    }

    #[test]
    fn team_includes_fainted_bench() {
        // 瀕死ベンチは `0 fnt` → hp=0, max_hp=0, fainted=true。
        let req = parse_request_json(
            r#"{"active":[{"moves":[{"id":"tackle","disabled":false}]}],"side":{"pokemon":[{"ident":"p2: Mew","condition":"40/175"},{"ident":"p2: Ditto","condition":"0 fnt"}]}}"#,
        );
        let Request::Move { team, .. } = req else {
            panic!("expected Move request");
        };
        assert_eq!(
            team,
            vec![
                TeamMember { mon: PokemonRef::new(2, "Mew"), hp: 40, max_hp: 175, fainted: false },
                TeamMember { mon: PokemonRef::new(2, "Ditto"), hp: 0, max_hp: 0, fainted: true },
            ]
        );
    }

    #[test]
    fn faint_and_win_lines() {
        assert_eq!(
            parse_event_line("|faint|p1a: Mew"),
            Some(Event::Faint { target: PokemonRef::new(1, "Mew") })
        );
        assert_eq!(
            parse_event_line("|win|P1"),
            Some(Event::Win { player: "P1".to_string() })
        );
    }
}
