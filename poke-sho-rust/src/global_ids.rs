//! 種族・技・タイプのグローバル ID 表。
//!
//! `data/*.tsv` は `scripts/gen_global_ids.mjs` で pokemon-showdown の dex から
//! 一度だけ生成したもので、**ID は永久に不変** (Embedding の行番号として
//! checkpoint に焼き付くため)。追加はファイル末尾への追記のみ許される。
//! 名前は Showdown の正式表記 (フォルム込み) と一致する。

use std::collections::HashMap;
use std::sync::OnceLock;

const SPECIES_TSV: &str = include_str!("../data/species_ids.tsv");
const MOVE_TSV: &str = include_str!("../data/move_ids.tsv");
const TYPE_TSV: &str = include_str!("../data/type_ids.tsv");

/// Embedding テーブルの固定容量 (語彙の上限)。実 TSV 長 (`vocab_sizes`) はこれ以下で
/// なければならない (`tables()` で検証)。新種族・新技を TSV 末尾へ追記しても、ID は
/// append-only で不変・テーブル形状はこの容量で固定されるため checkpoint 互換が保たれる。
/// 一度決めたら下げない (上げると既存 checkpoint と shape mismatch になる)。
pub const SPECIES_VOCAB_CAP: usize = 2048;
pub const MOVE_VOCAB_CAP: usize = 2048;

/// 技カテゴリ (Embedding ではなくスカラ特徴として使う)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveCategory {
    Physical = 0,
    Special = 1,
    Status = 2,
}

/// 技 1 件のグローバルメタデータ。
#[derive(Debug, Clone, Copy)]
pub struct GlobalMoveMeta {
    pub id: u16,
    pub type_id: u8,
    pub category: MoveCategory,
    pub base_power: u16,
}

/// 種族 1 件のグローバルメタデータ。`type2` は単タイプなら `None`。
#[derive(Debug, Clone, Copy)]
pub struct GlobalSpeciesMeta {
    pub id: u16,
    pub type1: u8,
    pub type2: Option<u8>,
}

struct Tables {
    species: HashMap<&'static str, GlobalSpeciesMeta>,
    moves: HashMap<&'static str, GlobalMoveMeta>,
    types: HashMap<&'static str, u8>,
    species_len: usize,
    moves_len: usize,
    types_len: usize,
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut types = HashMap::new();
        for line in TYPE_TSV.lines().skip(1) {
            let mut it = line.split('\t');
            let id: u8 = it.next().unwrap().parse().unwrap();
            let name = it.next().unwrap();
            types.insert(name, id);
        }
        let mut species = HashMap::new();
        for line in SPECIES_TSV.lines().skip(1) {
            let mut it = line.split('\t');
            let id: u16 = it.next().unwrap().parse().unwrap();
            let name = it.next().unwrap();
            let _dex_num = it.next().unwrap();
            let mut ts = it.next().unwrap().split('/');
            let type1 = types[ts.next().unwrap()];
            let type2 = ts.next().map(|t| types[t]);
            species.insert(name, GlobalSpeciesMeta { id, type1, type2 });
        }
        let mut moves = HashMap::new();
        for line in MOVE_TSV.lines().skip(1) {
            let mut it = line.split('\t');
            let id: u16 = it.next().unwrap().parse().unwrap();
            let name = it.next().unwrap();
            let type_id = types[it.next().unwrap()];
            let category = match it.next().unwrap() {
                "Physical" => MoveCategory::Physical,
                "Special" => MoveCategory::Special,
                _ => MoveCategory::Status,
            };
            let base_power: u16 = it.next().unwrap().parse().unwrap();
            moves.insert(name, GlobalMoveMeta { id, type_id, category, base_power });
        }
        assert!(
            species.len() <= SPECIES_VOCAB_CAP,
            "species vocab {} exceeds SPECIES_VOCAB_CAP {SPECIES_VOCAB_CAP}; raise the cap (breaks checkpoint compat)",
            species.len()
        );
        assert!(
            moves.len() <= MOVE_VOCAB_CAP,
            "move vocab {} exceeds MOVE_VOCAB_CAP {MOVE_VOCAB_CAP}; raise the cap (breaks checkpoint compat)",
            moves.len()
        );
        Tables {
            species_len: species.len(),
            moves_len: moves.len(),
            types_len: types.len(),
            species,
            moves,
            types,
        }
    })
}

/// Showdown 種族名 (フォルム込み) → グローバルメタデータ。
pub fn species_meta(name: &str) -> Option<GlobalSpeciesMeta> {
    tables().species.get(name).copied()
}

/// Showdown 技名 → グローバルメタデータ。
pub fn move_meta(name: &str) -> Option<GlobalMoveMeta> {
    tables().moves.get(name).copied()
}

/// タイプ名 → タイプ ID。
pub fn type_id(name: &str) -> Option<u8> {
    tables().types.get(name).copied()
}

/// Embedding テーブルの語彙サイズ (種族, 技, タイプ)。
pub fn vocab_sizes() -> (usize, usize, usize) {
    let t = tables();
    (t.species_len, t.moves_len, t.types_len)
}

/// 生 TSV (Python 側がそのままパースする。フォーマット: ヘッダ行 + タブ区切り)。
pub fn species_tsv() -> &'static str {
    SPECIES_TSV
}

pub fn move_tsv() -> &'static str {
    MOVE_TSV
}

pub fn type_tsv() -> &'static str {
    TYPE_TSV
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups_match_generated_table() {
        let cloyster = species_meta("Cloyster").unwrap();
        assert_eq!(cloyster.id, 149);
        assert_eq!(cloyster.type1, type_id("Water").unwrap());
        assert_eq!(cloyster.type2, Some(type_id("Ice").unwrap()));

        let goodra_h = species_meta("Goodra-Hisui").unwrap();
        assert_eq!(goodra_h.id, 957);
        let goodra = species_meta("Goodra").unwrap();
        assert_eq!(goodra.id, 956);
        assert_eq!(goodra.type2, None);

        let crunch = move_meta("Crunch").unwrap();
        assert_eq!(crunch.id, 150);
        assert_eq!(crunch.category, MoveCategory::Physical);
        assert_eq!(crunch.base_power, 80);
        assert_eq!(crunch.type_id, type_id("Dark").unwrap());

        // stage3c の合成技 (move_ids.tsv 末尾に追加)。
        let fight = move_meta("FightSpe60").unwrap();
        assert_eq!(fight.id, 951);
        assert_eq!(fight.category, MoveCategory::Special);
        assert_eq!(fight.base_power, 60);
        assert_eq!(fight.type_id, type_id("Fighting").unwrap());
        let fairy = move_meta("FairyPhy60").unwrap();
        assert_eq!(fairy.id, 952);
        assert_eq!(fairy.category, MoveCategory::Physical);
        assert_eq!(fairy.type_id, type_id("Fairy").unwrap());

        let (ns, nm, nt) = vocab_sizes();
        assert_eq!(ns, 1417);
        assert_eq!(nm, 953);
        assert_eq!(nt, 19);
    }
}
