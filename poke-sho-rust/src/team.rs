//! Parser for the Pokemon Showdown team export text format.
//!
//! Supports the subset needed by the early phases: species, optional nickname /
//! item / ability / tera type / level, nature, EVs, IVs, and a move list.
//! Unspecified IVs default to 31, EVs to 0, and—because these phases are fixed
//! at level 50 and the export omits a `Level:` line—the level defaults to 50.

use crate::moves::{MoveData, move_by_name};
use crate::species::{Nature, Species, Stats, compute_stats, species_by_name};

/// Default level for sets that omit a `Level:` line (phase convention).
pub const DEFAULT_LEVEL: u16 = 50;

/// Error produced while parsing or resolving a team.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeamError {
    Empty,
    UnknownSpecies(String),
    UnknownMove(String),
    UnknownNature(String),
    BadStatLine(String),
}

/// A single parsed set, before species/move resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSet {
    pub species: String,
    pub nickname: Option<String>,
    pub item: Option<String>,
    pub ability: Option<String>,
    pub level: u16,
    pub tera_type: Option<String>,
    pub nature: Nature,
    pub evs: Stats,
    pub ivs: Stats,
    pub moves: Vec<String>,
}

/// A set with its species, real stats, and moves resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSet {
    pub species: Species,
    pub nickname: Option<String>,
    pub level: u16,
    pub nature: Nature,
    pub stats: Stats,
    pub moves: Vec<MoveData>,
}

fn parse_species_line(line: &str) -> (String, Option<String>, Option<String>) {
    // Optional " @ Item" suffix.
    let (head, item) = match line.split_once('@') {
        Some((h, i)) => (h.trim(), Some(i.trim().to_string())),
        None => (line.trim(), None),
    };
    // Optional "Nickname (Species)" form.
    if let Some(open) = head.rfind('(') {
        if head.ends_with(')') {
            let nickname = head[..open].trim().to_string();
            let species = head[open + 1..head.len() - 1].trim().to_string();
            return (species, Some(nickname), item);
        }
    }
    (head.to_string(), None, item)
}

fn apply_stat_line(base: Stats, value: &str) -> Result<Stats, TeamError> {
    let mut stats = base;
    for part in value.split('/') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (num, name) = part
            .split_once(char::is_whitespace)
            .ok_or_else(|| TeamError::BadStatLine(part.to_string()))?;
        let n: u16 = num
            .trim()
            .parse()
            .map_err(|_| TeamError::BadStatLine(part.to_string()))?;
        match name.trim().to_ascii_lowercase().as_str() {
            "hp" => stats.hp = n,
            "atk" => stats.atk = n,
            "def" => stats.def = n,
            "spa" => stats.spa = n,
            "spd" => stats.spd = n,
            "spe" => stats.spe = n,
            _ => return Err(TeamError::BadStatLine(part.to_string())),
        }
    }
    Ok(stats)
}

/// Parses one set from its non-empty lines.
fn parse_set(lines: &[&str]) -> Result<ParsedSet, TeamError> {
    let mut iter = lines.iter();
    let first = iter.next().ok_or(TeamError::Empty)?;
    let (species, nickname, item) = parse_species_line(first);

    let mut set = ParsedSet {
        species,
        nickname,
        item,
        ability: None,
        level: DEFAULT_LEVEL,
        tera_type: None,
        nature: Nature::Serious,
        evs: Stats::ZERO,
        ivs: Stats::DEFAULT_IVS,
        moves: Vec::new(),
    };

    for line in iter {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('-') {
            set.moves.push(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("Ability:") {
            set.ability = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("Level:") {
            set.level = rest
                .trim()
                .parse()
                .map_err(|_| TeamError::BadStatLine(line.to_string()))?;
        } else if let Some(rest) = line.strip_prefix("Tera Type:") {
            set.tera_type = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("EVs:") {
            set.evs = apply_stat_line(Stats::ZERO, rest)?;
        } else if let Some(rest) = line.strip_prefix("IVs:") {
            set.ivs = apply_stat_line(Stats::DEFAULT_IVS, rest)?;
        } else if let Some(name) = line.strip_suffix("Nature") {
            set.nature = Nature::from_name(name.trim())
                .ok_or_else(|| TeamError::UnknownNature(name.trim().to_string()))?;
        }
        // Other lines (Shiny, Happiness, etc.) are ignored for now.
    }

    Ok(set)
}

/// Parses a full team export into its sets.
pub fn parse_team(text: &str) -> Result<Vec<ParsedSet>, TeamError> {
    let mut sets = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            if !current.is_empty() {
                sets.push(parse_set(&current)?);
                current.clear();
            }
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        sets.push(parse_set(&current)?);
    }
    if sets.is_empty() {
        return Err(TeamError::Empty);
    }
    Ok(sets)
}

/// Resolves a parsed set into its species, real stats, and move data.
pub fn resolve_set(set: &ParsedSet) -> Result<ResolvedSet, TeamError> {
    let species = species_by_name(&set.species)
        .ok_or_else(|| TeamError::UnknownSpecies(set.species.clone()))?;
    let stats = compute_stats(species.base, set.ivs, set.evs, set.level, set.nature);
    let mut moves = Vec::with_capacity(set.moves.len());
    for name in &set.moves {
        let mv = move_by_name(name).ok_or_else(|| TeamError::UnknownMove(name.clone()))?;
        moves.push(mv);
    }
    Ok(ResolvedSet {
        species,
        nickname: set.nickname.clone(),
        level: set.level,
        nature: set.nature,
        stats,
        moves,
    })
}

/// Parses the first set of a team text and resolves it.
pub fn resolve_first(text: &str) -> Result<ResolvedSet, TeamError> {
    let sets = parse_team(text)?;
    resolve_set(&sets[0])
}

/// Parses and resolves every set of a team text (party order preserved).
pub fn resolve_team(text: &str) -> Result<Vec<ResolvedSet>, TeamError> {
    parse_team(text)?.iter().map(resolve_set).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::{CRUNCH, DARK_PULSE};

    const CLOYSTER_TEAM: &str = "Cloyster @ Covert Cloak\nAbility: Skill Link\nLevel: 50\nTera Type: Water\nRelaxed Nature\n- Crunch\n- Dark Pulse";

    #[test]
    fn parses_cloyster_team() {
        let sets = parse_team(CLOYSTER_TEAM).unwrap();
        assert_eq!(sets.len(), 1);
        let set = &sets[0];
        assert_eq!(set.species, "Cloyster");
        assert_eq!(set.item.as_deref(), Some("Covert Cloak"));
        assert_eq!(set.ability.as_deref(), Some("Skill Link"));
        assert_eq!(set.tera_type.as_deref(), Some("Water"));
        assert_eq!(set.nature, Nature::Relaxed);
        assert_eq!(set.level, DEFAULT_LEVEL);
        assert_eq!(set.moves, vec!["Crunch", "Dark Pulse"]);
    }

    #[test]
    fn resolves_cloyster_team() {
        let resolved = resolve_first(CLOYSTER_TEAM).unwrap();
        assert_eq!(resolved.species.name, "Cloyster");
        // Relaxed (+Def/-Spe) raises Def by 10%, lowers Spe by 10%; others neutral.
        assert_eq!(resolved.stats.def, 220);
        assert_eq!(resolved.stats.spe, 81);
        assert_eq!(resolved.moves, vec![CRUNCH, DARK_PULSE]);
    }

    #[test]
    fn parses_nickname_item_and_evs() {
        let text = "Shelly (Cloyster) @ Covert Cloak\nLevel: 50\nAdamant Nature\nEVs: 252 Atk / 4 Def\nIVs: 0 Spe\n- Crunch";
        let set = &parse_team(text).unwrap()[0];
        assert_eq!(set.species, "Cloyster");
        assert_eq!(set.nickname.as_deref(), Some("Shelly"));
        assert_eq!(set.item.as_deref(), Some("Covert Cloak"));
        assert_eq!(set.evs.atk, 252);
        assert_eq!(set.evs.def, 4);
        assert_eq!(set.ivs.spe, 0);
        assert_eq!(set.ivs.hp, 31);
        assert_eq!(set.nature, Nature::Adamant);
    }

    #[test]
    fn stage3_cloyster_mirror_222_94() {
        let text = "Cloyster @ Covert Cloak\nLevel: 50\nEVs: 252 HP / 12 Atk / 12 Def / 92 SpA / 228 SpD / 76 Spe\nRelaxed Nature\n- Crunch";
        let s = resolve_first(text).unwrap().stats;
        // 鏡像: HP157 / A117 / B222 / C117 / D94 / S90.
        assert_eq!((s.hp, s.atk, s.def, s.spa, s.spd, s.spe), (157, 117, 222, 117, 94, 90));
    }

    #[test]
    fn stage3_goodra_hisui_mirror_94_222() {
        let text = "Goodra-Hisui @ Covert Cloak\nLevel: 50\nEVs: 12 HP / 252 SpD / 76 Spe\nIVs: 0 Def / 24 Atk / 4 SpA\nGentle Nature\n- Crunch";
        let r = resolve_first(text).unwrap();
        let s = r.stats;
        assert_eq!(r.species.name, "Goodra-Hisui");
        // 鏡像: HP157 / A117 / B94 / C117 / D222 / S90.
        assert_eq!((s.hp, s.atk, s.def, s.spa, s.spd, s.spe), (157, 117, 94, 117, 222, 90));
    }

    #[test]
    fn unknown_species_errors() {
        let err = resolve_first("Pikachu\n- Crunch").unwrap_err();
        assert_eq!(err, TeamError::UnknownSpecies("Pikachu".to_string()));
    }
}
