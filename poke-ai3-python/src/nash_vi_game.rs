//! Nash 値反復 (Shapley 確率ゲーム) 用のゲーム定義。
//!
//! 状態空間は `policy_table.rs` と**完全に同じ dense index** (cross-team, HP を
//! `hp_buckets` 段に離散化, radix: ai_team[2], ai_active[2], ai_hp_c[H], ai_hp_g[H],
//! opp_active[2], opp_hp_c[H], opp_hp_g[H], 総数 8·H^4)。AI=P1・相手=P2 で構築する。
//!
//! ここは「dense index ⇄ BattleState」「バケット代表 HP ⇄ バケット」「合法手取得」の
//! 純粋ヘルパに徹する。遷移分布は `nash_vi_trans`、行列ゲーム解法と VI は `nash_vi_solve`。

use poke_sho_rust::battle::{BattleState, Choice, Player};
use poke_sho_rust::scenario::{Stage, TeamId};

fn team_of(v: u64) -> TeamId {
    if v == 0 { TeamId::Team1 } else { TeamId::Team2 }
}

/// バケット index → 代表 HP。k=0→0(瀕死), k=H-1→満タン。policy_table.rs と同一。
pub fn bucket_to_hp(k: u64, max_hp: i32, buckets: u64) -> i32 {
    if buckets <= 1 {
        return max_hp;
    }
    ((k as f64) * (max_hp as f64) / ((buckets - 1) as f64)).round() as i32
}

/// 実 HP → 最近傍バケット (bucket_to_hp の逆写像, runtime JS と一致)。
pub fn hp_to_bucket(hp: i32, max_hp: i32, buckets: u64) -> u64 {
    if buckets <= 1 || max_hp <= 0 {
        return 0;
    }
    let hp = hp.max(0);
    let b = ((hp as f64) * ((buckets - 1) as f64) / (max_hp as f64)).round() as i64;
    b.clamp(0, buckets as i64 - 1) as u64
}

/// dense index を分解した状態次元。member 0=Cloyster, 1=Goodra(宣言順)。
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub ai_team: u64,
    pub ai_active: u64,
    pub ai_hp_c: u64,
    pub ai_hp_g: u64,
    pub opp_active: u64,
    pub opp_hp_c: u64,
    pub opp_hp_g: u64,
}

impl Dims {
    /// 密 index を分解 (policy_table.rs と同じ最下位→最上位の並び)。
    pub fn decompose(k: u64, h: u64) -> Self {
        let mut r = k;
        let opp_hp_g = r % h; r /= h;
        let opp_hp_c = r % h; r /= h;
        let opp_active = r % 2; r /= 2;
        let ai_hp_g = r % h; r /= h;
        let ai_hp_c = r % h; r /= h;
        let ai_active = r % 2; r /= 2;
        let ai_team = r % 2;
        Dims { ai_team, ai_active, ai_hp_c, ai_hp_g, opp_active, opp_hp_c, opp_hp_g }
    }

    /// 状態次元 → 密 index (decompose の逆)。
    pub fn compose(&self, h: u64) -> u64 {
        let mut k = self.ai_team;
        k = k * 2 + self.ai_active;
        k = k * h + self.ai_hp_c;
        k = k * h + self.ai_hp_g;
        k = k * 2 + self.opp_active;
        k = k * h + self.opp_hp_c;
        k = k * h + self.opp_hp_g;
        k
    }

    /// P1↔P2 を入れ替えた鏡像状態 (相手の均衡戦略を対称性で引くのに使う)。
    /// opp_team = 1 - ai_team なので ai_team も反転する。
    pub fn swap(&self) -> Self {
        Dims {
            ai_team: 1 - self.ai_team,
            ai_active: self.opp_active,
            ai_hp_c: self.opp_hp_c,
            ai_hp_g: self.opp_hp_g,
            opp_active: self.ai_active,
            opp_hp_c: self.ai_hp_c,
            opp_hp_g: self.ai_hp_g,
        }
    }

    /// アクティブが瀕死 (bucket 0) になる組合せは実局面に現れない (= 無効/除外)。
    pub fn active_alive(&self) -> bool {
        let ai_b = if self.ai_active == 0 { self.ai_hp_c } else { self.ai_hp_g };
        let opp_b = if self.opp_active == 0 { self.opp_hp_c } else { self.opp_hp_g };
        ai_b != 0 && opp_b != 0
    }
}

/// dense index から BattleState を構築 (AI=P1・相手=P2、cross-team、HP はバケット代表値)。
/// policy_table.rs の enumerate_policy_batch と同一構築。
pub fn build_state(d: &Dims, h: u64, stage: Stage) -> BattleState {
    let opp_team = 1 - d.ai_team;
    let mut st = BattleState::new_with_teams(
        stage,
        (team_of(d.ai_team), d.ai_active as usize),
        (team_of(opp_team), d.opp_active as usize),
    );
    set_member_hp(&mut st, 0, 0, d.ai_hp_c, h);
    set_member_hp(&mut st, 0, 1, d.ai_hp_g, h);
    set_member_hp(&mut st, 1, 0, d.opp_hp_c, h);
    set_member_hp(&mut st, 1, 1, d.opp_hp_g, h);
    st
}

fn set_member_hp(st: &mut BattleState, side: usize, member: usize, bucket: u64, buckets: u64) {
    let mon = &mut st.parties[side].members[member];
    mon.hp = bucket_to_hp(bucket, mon.max_hp, buckets);
}

/// ai_team ごとの満タン雛形 (team text のパースは高価なので事前構築し、以降は
/// clone + HP/active 差し替えだけで各状態を作る)。index=ai_team。
pub fn templates(stage: Stage) -> [BattleState; 2] {
    [
        BattleState::new_with_teams(stage, (team_of(0), 0), (team_of(1), 0)),
        BattleState::new_with_teams(stage, (team_of(1), 0), (team_of(0), 0)),
    ]
}

/// 雛形 (対応する ai_team) を clone し、active と 4 体の HP をバケット代表値へ。
pub fn fill(base: &BattleState, d: &Dims, h: u64) -> BattleState {
    let mut st = *base;
    st.parties[0].active = d.ai_active as usize;
    st.parties[1].active = d.opp_active as usize;
    set_member_hp(&mut st, 0, 0, d.ai_hp_c, h);
    set_member_hp(&mut st, 0, 1, d.ai_hp_g, h);
    set_member_hp(&mut st, 1, 0, d.opp_hp_c, h);
    set_member_hp(&mut st, 1, 1, d.opp_hp_g, h);
    st
}

/// 100 手到達 (無限 stall) 時の終端値 (P1 視点)。手数制限は状態に無いが、実ゲームは
/// 100 手で打ち切り。draw(0.5) だと「交代し続けて引分逃げ」が安全手になり定常不動点が
/// 縮退する。代わりに残存で優劣を採点する: 生存数を優先し、同数なら HP 割合和で比較。
/// これで終端値が (turn 非依存の) 状態だけで確定し、backward induction が「有利側は最終手に
/// 攻撃して勝つ」を後ろ向きに unravel できる (stall は均衡から消える)。
pub fn tiebreak(d: &Dims, h: u64) -> f32 {
    let frac = |b: u64| if h <= 1 { 1.0 } else { b as f32 / (h - 1) as f32 };
    let p1_alive = (d.ai_hp_c > 0) as u32 + (d.ai_hp_g > 0) as u32;
    let p2_alive = (d.opp_hp_c > 0) as u32 + (d.opp_hp_g > 0) as u32;
    if p1_alive != p2_alive {
        return if p1_alive > p2_alive { 1.0 } else { 0.0 };
    }
    let p1 = frac(d.ai_hp_c) + frac(d.ai_hp_g);
    let p2 = frac(d.opp_hp_c) + frac(d.opp_hp_g);
    if (p1 - p2).abs() < 1e-6 {
        0.5
    } else if p1 > p2 {
        1.0
    } else {
        0.0
    }
}

/// 遷移後 BattleState の帰結。ai_team は不変なので引数で渡す (state から復元しない)。
pub enum Outcome {
    /// 生存継続。次状態の dense index。
    Index(u64),
    /// 終局。P1(AI) 視点の値 (勝ち=1.0 / 負け=0.0 / 相打ち引分=0.5)。
    Terminal(f32),
}

/// 遷移後の BattleState を Outcome へ (index 化 or 終局判定)。ai_team は不変値を渡す。
pub fn classify(st: &BattleState, ai_team: u64, h: u64) -> Outcome {
    let p1_lost = st.parties[0].all_fainted();
    let p2_lost = st.parties[1].all_fainted();
    match (p1_lost, p2_lost) {
        (false, true) => return Outcome::Terminal(1.0),
        (true, false) => return Outcome::Terminal(0.0),
        (true, true) => return Outcome::Terminal(0.5),
        (false, false) => {}
    }
    // hp>0 の生存個体は必ず bucket>=1 に落とす (量子化で微小 HP が bucket0=瀕死に丸まると、
    // active_alive()==false の無効 index になり継続値が抜ける → 鏡像対称が破れる)。bucket0 は
    // 瀕死 (hp<=0) 専用。これで列挙 (bucket0⟺HP0) と classify の意味論が一致する。
    let bkt = |hp: i32, max: i32| -> u64 {
        if hp <= 0 { 0 } else { hp_to_bucket(hp, max, h).max(1) }
    };
    let d = Dims {
        ai_team,
        ai_active: st.parties[0].active as u64,
        ai_hp_c: bkt(st.parties[0].members[0].hp, st.parties[0].members[0].max_hp),
        ai_hp_g: bkt(st.parties[0].members[1].hp, st.parties[0].members[1].max_hp),
        opp_active: st.parties[1].active as u64,
        opp_hp_c: bkt(st.parties[1].members[0].hp, st.parties[1].members[0].max_hp),
        opp_hp_g: bkt(st.parties[1].members[1].hp, st.parties[1].members[1].max_hp),
    };
    Outcome::Index(d.compose(h))
}

/// 合法手 (技/交代) を取得。強制交代でない通常決定点を仮定。
pub fn legal(st: &BattleState, player: Player) -> Vec<Choice> {
    st.legal_choices(player)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_decompose_roundtrip() {
        let h = 26;
        for k in [0u64, 1, 12345, 8 * h * h * h * h - 1] {
            let d = Dims::decompose(k, h);
            assert_eq!(d.compose(h), k);
        }
    }

    #[test]
    fn bucket_roundtrip_reps() {
        // 代表 HP はそのバケットへ戻る。
        let (h, max) = (26u64, 200i32);
        for k in 0..h {
            let hp = bucket_to_hp(k, max, h);
            assert_eq!(hp_to_bucket(hp, max, h), k, "bucket {k} hp {hp}");
        }
    }

    #[test]
    fn build_state_matches_index() {
        // 構築 → classify で同じ index に戻る (有効状態)。
        let h = 26;
        let d = Dims { ai_team: 1, ai_active: 0, ai_hp_c: 25, ai_hp_g: 10,
                       opp_active: 1, opp_hp_c: 7, opp_hp_g: 25 };
        let st = build_state(&d, h, Stage::Stage3b);
        match classify(&st, d.ai_team, h) {
            Outcome::Index(k) => assert_eq!(k, d.compose(h)),
            Outcome::Terminal(_) => panic!("should be alive"),
        }
    }
}
