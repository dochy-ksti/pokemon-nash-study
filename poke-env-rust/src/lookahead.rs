//! lookahead 探索 (poke-poke オセロ算法の移植)。
//!
//! 各局面で、ルートの各候補手を起点に policy に沿って終局まで rollout し、
//! その平均勝率を求める。最初の一手だけ候補手で固定し、それ以外 (相手の同手番の手と、
//! 以降の両者の手) は policy net からサンプルする。深さ上限に達して終局しない場合のみ
//! value net で末端を評価する (1v1 は短いので通常は終局に到達する)。
//!
//! 得られた手ごとの平均勝率から、`nash::nash_accumulation` で policy の教師
//! (`training_pi`) と着手用分布 (`selection_pi`) を作り、value の教師 (= 手ごと最大勝率)
//! を返す。policy/value の推論だけを `PolicyOracle` 経由で外部に委譲する。

use crate::nash::nash_accumulation;
use crate::observation::{ACTION_DIM, Party, action_index, observation_for};
use crate::oracle::{OracleOut, PolicyOracle};
use crate::battle_chacha::BattleChaCha;
use poke_sho_rust::battle::{BattleState, Choice, Player, apply_forced_switches, apply_turn};

/// lookahead のハイパーパラメータ。
#[derive(Debug, Clone, Copy)]
pub struct LookaheadConfig {
    /// 1 局面あたりの rollout 回数。
    pub sims: u32,
    /// 1 局面の lookahead 内で同時に in-flight にする rollout の本数 (スライディング
    /// ウィンドウ幅)。常にこの本数を維持し、1 本完了するたびに次を起動する。1 で従来の
    /// 逐次挙動。`sims` を超えてはならない (呼び出し側で検証する)。
    pub sim_concurrency: u32,
    /// rollout の最小・最大深さ (ply)。終局はこれより手前で来ることが多い。
    pub search_turn_min: u32,
    pub search_turn_max: u32,
    /// depth_cap を [min..=max] から選ぶときの深側への偏り。深さ k 番目 (min が 0) の
    /// 選択重みを `depth_skew^k` とする。1.0 で一様、2.0 で「1手深いごとに確率2倍」。
    /// 行動分布には一切触れず、value 推定の打ち切り地平線の配分だけを変えるため Nash-safe。
    pub depth_skew: f32,
    /// true で lookahead (value rollout) を完全に廃し、各手番で policy net を1回だけ
    /// 推論して確率サンプルで着手する高速対戦モード。checkpoint 同士の強さ判定用。
    /// 行動分布は素の policy のまま (Nash 混合戦略を保つ)。
    pub policy_only: bool,
    pub nash_learning_rate: f32,
    pub nash_minimum_pi: f32,
    pub nash_pi_limit: f32,
    /// true で nash_accumulation の穏当化版 (崖なし) を使う。デフォルトは true。
    /// strict 版 (崖あり) は 2 seed の A/B で weak と同等以下だったため非採用。
    /// しばらく様子見で残すが、採用機会が無ければ削除予定。
    pub nash_weak: bool,
    pub cutoff_coeff: f32,
    /// rollout 中の 16 段ダメージ乱数・急所 (本番と同じ設定にする)。
    pub randomize: bool,
    pub crit_enabled: bool,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            sims: 64,
            sim_concurrency: 1,
            search_turn_min: 4,
            search_turn_max: 8,
            depth_skew: 1.0,
            policy_only: false,
            nash_learning_rate: 2.0,
            nash_minimum_pi: 0.03,
            nash_pi_limit: 0.05,
            nash_weak: true,
            cutoff_coeff: 1.0,
            randomize: false,
            crit_enabled: false,
        }
    }
}

/// lookahead の出力。学習サンプルの (π, V) 教師と、実対局を進めるための分布。
/// π は長さ `ACTION_DIM` (技 + 交代枠、交代は相対控え index)。
#[derive(Debug, Clone, Copy)]
pub struct LookaheadResult {
    pub training_pi: [f32; ACTION_DIM],
    pub selection_pi: [f32; ACTION_DIM],
    pub value: f32,
    /// 各合法手の rollout 平均勝率 (root 視点 0..1)。非合法手・未試行は 0.0。
    /// 「無差別局面 (手で勝率が変わらない) かどうか」を診断するために公開する。
    pub win_rates: [f32; ACTION_DIM],
}

/// policy 確率から legal 手に制限して 1 つ選ぶ。技はスロット相対、交代は相対控え index。
fn sample_choice(
    policy: &[f32; ACTION_DIM],
    legal: &[Choice],
    party: &Party,
    rng: &mut BattleChaCha,
) -> Choice {
    let mut total = 0.0f32;
    for c in legal {
        total += policy[action_index(party, *c)].max(0.0);
    }
    if total <= 0.0 {
        return legal[0];
    }
    let r = rng.unit() * total;
    let mut acc = 0.0f32;
    for c in legal {
        acc += policy[action_index(party, *c)].max(0.0);
        if r < acc {
            return *c;
        }
    }
    *legal.last().unwrap()
}

/// rollout 中に瀕死後の強制交代サブフェーズを解決する。強制側だけ policy から交代手を
/// サンプルし `apply_forced_switches` で場へ戻す (local_showdown と同じ部品)。
async fn resolve_forced_switches<O: PolicyOracle>(
    oracle: &O,
    state: &mut BattleState,
    rng: &mut BattleChaCha,
) {
    while !state.is_done() && state.any_forced_switch() {
        let need1 = state.needs_forced_switch(Player::P1);
        let need2 = state.needs_forced_switch(Player::P2);
        let (o1, o2) = tokio::join!(
            infer_forced(oracle, state, Player::P1, need1),
            infer_forced(oracle, state, Player::P2, need2),
        );
        let c1 = o1.map(|out| {
            sample_choice(
                &out.policy,
                &state.legal_choices(Player::P1),
                state.party(Player::P1),
                rng,
            )
        });
        let c2 = o2.map(|out| {
            sample_choice(
                &out.policy,
                &state.legal_choices(Player::P2),
                state.party(Player::P2),
                rng,
            )
        });
        *state = apply_forced_switches(*state, c1, c2).state;
    }
}

/// 強制交代待ちの側だけ policy/value を推論する。待っていない側は `None`。
async fn infer_forced<O: PolicyOracle>(
    oracle: &O,
    state: &BattleState,
    player: Player,
    needs: bool,
) -> Option<OracleOut> {
    if !needs {
        return None;
    }
    Some(oracle.infer(observation_for(state, player), player).await)
}

/// rollout の打ち切り深さ `depth_cap` を [min..=max] から重み付きで1つ選ぶ。
/// 深さ k 番目 (min を 0 とする) の重みを `depth_skew^k` とし、skew>1 で深側に偏らせる。
/// skew<=0 や非有限値、span=1 のときは一様 (= 従来挙動) にフォールバックする。
fn sample_depth_cap(cfg: &LookaheadConfig, rng: &mut BattleChaCha) -> u32 {
    let min = cfg.search_turn_min;
    let max = cfg.search_turn_max.max(min);
    let span = max - min + 1;
    let skew = cfg.depth_skew;
    if span == 1 || !(skew.is_finite() && skew > 0.0) || (skew - 1.0).abs() < 1e-6 {
        return min + rng.below(span);
    }
    let mut weights = Vec::with_capacity(span as usize);
    let mut total = 0.0f32;
    let mut w = 1.0f32;
    for _ in 0..span {
        weights.push(w);
        total += w;
        w *= skew;
    }
    let r = rng.unit() * total;
    let mut acc = 0.0f32;
    for (k, weight) in weights.iter().enumerate() {
        acc += weight;
        if r < acc {
            return min + k as u32;
        }
    }
    max
}

/// 候補手 `forced` を root の最初の一手として固定し、終局まで rollout する。
/// root 視点の勝率 (0.0..1.0) を返す。
async fn rollout<O: PolicyOracle>(
    oracle: &O,
    start: BattleState,
    root: Player,
    forced: Choice,
    seed: u64,
    cfg: &LookaheadConfig,
) -> f32 {
    let mut rng = BattleChaCha::from_u64(seed, cfg.randomize, cfg.crit_enabled);
    let depth_cap = sample_depth_cap(cfg, &mut rng);
    let mut state = start;
    let mut ply = 0u32;
    loop {
        if state.is_done() {
            // 引き分け (ターン上限) は中立 0.5。勝敗決着は root 視点で 1.0 / 0.0。
            return match state.winner() {
                Some(w) => {
                    if w == root {
                        1.0
                    } else {
                        0.0
                    }
                }
                None => 0.5,
            };
        }
        // GPU を使う infer は P1/P2 を同時に poll させ、同一バッチに乗りやすくする。
        // rng を使う sampling は infer 不要なので後段で逐次に行い、`&mut rng` の衝突を避ける。
        let (p1_out, p2_out) = tokio::join!(
            infer_move(oracle, &state, Player::P1, root, ply),
            infer_move(oracle, &state, Player::P2, root, ply),
        );
        let p1_choice = match p1_out {
            Some(out) => sample_choice(
                &out.policy,
                &state.legal_choices(Player::P1),
                state.party(Player::P1),
                &mut rng,
            ),
            None => forced,
        };
        let p2_choice = match p2_out {
            Some(out) => sample_choice(
                &out.policy,
                &state.legal_choices(Player::P2),
                state.party(Player::P2),
                &mut rng,
            ),
            None => forced,
        };
        let first = rng.first_player();
        let res = apply_turn(state, p1_choice, p2_choice, first, &mut rng);
        state = res.state;
        // 瀕死後の強制交代サブフェーズを解決してから次 ply へ。
        resolve_forced_switches(oracle, &mut state, &mut rng).await;
        ply += 1;
        if state.is_done() {
            // 引き分け (ターン上限) は中立 0.5。勝敗決着は root 視点で 1.0 / 0.0。
            return match state.winner() {
                Some(w) => {
                    if w == root {
                        1.0
                    } else {
                        0.0
                    }
                }
                None => 0.5,
            };
        }
        if ply >= depth_cap {
            // 深さ上限で終局せず — value net で末端を評価 (root 視点の勝率 0..1)。
            let out = oracle.infer(observation_for(&state, root), root).await;
            return out.value.clamp(0.0, 1.0);
        }
    }
}

/// `player` の手の policy/value を推論する。root の最初の一手 (forced) は推論不要なので
/// `None` を返し、GPU 呼び出しを省く。返した `OracleOut` は呼び出し側で rng サンプリングする。
async fn infer_move<O: PolicyOracle>(
    oracle: &O,
    state: &BattleState,
    player: Player,
    root: Player,
    ply: u32,
) -> Option<OracleOut> {
    if player == root && ply == 0 {
        return None;
    }
    Some(oracle.infer(observation_for(state, player), player).await)
}

/// UCB1 (poke-poke と同じ式) で次に評価する arm を選ぶ。
///
/// 並列 rollout に対応するため次の扱いをする:
/// - `counts` は in-flight (起動済み・未完了) を含む dispatch 済み本数。起動時に加算する。
/// - `avg` は完了済み rollout だけの平均勝率。まだ 1 本も完了していない arm は `None` で、
///   活用項には provisional 0.5 を代入する。
/// - `counts[i] == 0` (一度も起動していない) の arm は探索項 +∞ で最優先する。起動した
///   瞬間に `counts` が加算されるので、最初の `legal` 本が各 arm に 1 本ずつ自然に割り当たる。
fn ucb_select(
    avg: &[Option<f32>],
    counts: &[u32],
    legal: &[Choice],
    party: &Party,
    total: u32,
) -> Choice {
    let mut best = legal[0];
    let mut best_u = f32::MIN;
    for c in legal {
        let i = action_index(party, *c);
        let u = if counts[i] == 0 {
            f32::INFINITY
        } else {
            let q = avg[i].unwrap_or(0.5);
            let cnt = counts[i] as f32;
            q + 2.0 * ((total + 1) as f32).ln() / cnt
        };
        if u > best_u {
            best_u = u;
            best = *c;
        }
    }
    best
}

/// 1 局面の lookahead を実行する。
pub async fn run_lookahead<O: PolicyOracle>(
    oracle: &O,
    start: BattleState,
    root: Player,
    rng_seed: u64,
    cfg: &LookaheadConfig,
) -> LookaheadResult {
    let root_party = start.party(root);
    let legal = start.legal_choices(root);

    // 合法手 1 つ (強制交代の単一控え等) なら rollout を打たず短絡即決する。
    // value は net 自身の予測を使う (分岐表 #9 の精神)。
    if legal.len() == 1 {
        let out = oracle.infer(observation_for(&start, root), root).await;
        let mut pi = [0.0f32; ACTION_DIM];
        pi[action_index(root_party, legal[0])] = 1.0;
        let v = out.value.clamp(0.0, 1.0);
        let mut win_rates = [0.0f32; ACTION_DIM];
        win_rates[action_index(root_party, legal[0])] = v;
        return LookaheadResult {
            training_pi: pi,
            selection_pi: pi,
            value: v,
            win_rates,
        };
    }

    // ルートの policy 予測 (Nash accumulation の起点)。rollout バーストと並行に投げ、
    // await は最後にまとめる。冒頭で単発 await すると、その間 pending 推論が 1 件に痩せて
    // バッチが細るため、join! で rollout と同時に poll させる。
    let predicted_fut = oracle.infer(observation_for(&start, root), root);

    // rollout のスライディングウィンドウ。win_rates と value を返す。
    let rollouts = async {
        // 1 スロットの仕事: real rollout か、穴埋め用 empty (ack 待ち)。
        enum SlotJob {
            Real { arm: Choice, idx: usize, seed: u64 },
            Empty,
        }
        enum SlotOutcome {
            Real { i: usize, win: f32 },
            Empty,
        }
        // real/empty を 1 つの async ブロックに包んで同一型にし、FuturesUnordered に混在させる。
        let run_slot = |job: SlotJob| async move {
            match job {
                SlotJob::Real { arm, idx, seed } => {
                    let win = rollout(oracle, start, root, arm, seed, cfg).await;
                    SlotOutcome::Real { i: idx, win }
                }
                SlotJob::Empty => {
                    oracle.ack_empty(root).await;
                    SlotOutcome::Empty
                }
            }
        };

        // `counts` は dispatch 済み (in-flight 含む) 本数、`sum_win`/`avg` は完了済みのみ。
        let mut sum_win = [0.0f32; ACTION_DIM];
        let mut counts = [0u32; ACTION_DIM];
        let mut done_counts = [0u32; ACTION_DIM];
        let mut avg: [Option<f32>; ACTION_DIM] = [None; ACTION_DIM];

        // スライディングウィンドウで常に `sim_concurrency` 本を維持する。real を出し切ったら
        // (dispatched == sims)、残りの real が完了するまで空きスロットを empty で埋める。
        // これにより各 lookahead が常に W 本を root へ計上し、threshold ゲートでデッドロックしない。
        let concurrency = cfg.sim_concurrency.max(1) as usize;
        let mut in_flight = futures::stream::FuturesUnordered::new();
        let mut dispatched = 0u32; // real rollout を dispatch した数
        let mut completed = 0u32; // real rollout が完了した数

        // ウィンドウを concurrency 本まで満たす。real 優先、無ければ (まだ real が残るなら) empty。
        macro_rules! refill {
            () => {
                while in_flight.len() < concurrency {
                    if dispatched < cfg.sims {
                        let total: u32 = counts.iter().sum();
                        let arm = ucb_select(&avg, &counts, &legal, root_party, total);
                        let idx = action_index(root_party, arm);
                        // dispatch カウンタを seed に使う (完了順は非決定的だが内容は決定的)。
                        let seed = rng_seed
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            .wrapping_add(dispatched as u64)
                            .wrapping_add((idx as u64) << 40);
                        counts[idx] += 1;
                        dispatched += 1;
                        in_flight.push(run_slot(SlotJob::Real { arm, idx, seed }));
                    } else if completed < cfg.sims {
                        in_flight.push(run_slot(SlotJob::Empty));
                    } else {
                        break;
                    }
                }
            };
        }

        refill!();
        while let Some(outcome) = futures::StreamExt::next(&mut in_flight).await {
            if let SlotOutcome::Real { i, win } = outcome {
                sum_win[i] += win;
                done_counts[i] += 1;
                completed += 1;
                avg[i] = Some(sum_win[i] / done_counts[i] as f32);
            }
            refill!();
        }

        let mut win_rates = [0.0f32; ACTION_DIM];
        let mut value = 0.0f32;
        for c in &legal {
            let i = action_index(root_party, *c);
            let wr = avg[i].unwrap_or(0.0);
            win_rates[i] = wr;
            if wr > value {
                value = wr;
            }
        }
        (win_rates, value)
    };

    let (predicted, (win_rates, value)) = tokio::join!(predicted_fut, rollouts);

    let mut legal_mask = [false; ACTION_DIM];
    for c in &legal {
        legal_mask[action_index(root_party, *c)] = true;
    }
    let (training_pi, selection_pi) =
        nash_accumulation(predicted.policy, legal_mask, win_rates, cfg);

    LookaheadResult {
        training_pi,
        selection_pi,
        value,
        win_rates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::StateForPlayer;
    use crate::oracle::OracleOut;
    use poke_sho_rust::scenario::{SpeciesId, Stage};

    /// 一様 policy・固定 value を返すモックオラクル。
    struct UniformOracle {
        value: f32,
    }

    impl PolicyOracle for UniformOracle {
        async fn infer(&self, _state: StateForPlayer, _player: Player) -> OracleOut {
            OracleOut {
                policy: [1.0 / ACTION_DIM as f32; ACTION_DIM],
                value: self.value,
            }
        }

        async fn ack_empty(&self, _player: Player) {}
    }

    /// rollout 方策が技を優先し自発交代しない oracle (交代スロットの確率を 0 にする)。
    /// 一様方策だと SE 交代の利得が乱数に埋もれるため、攻撃を撃ち合う方策の下で
    /// 「不利対面では控えへ交代した方が勝てる」という本来の傾向を検証するのに使う。
    struct AttackGreedyOracle {
        value: f32,
    }

    impl PolicyOracle for AttackGreedyOracle {
        async fn infer(&self, _state: StateForPlayer, _player: Player) -> OracleOut {
            let n = crate::observation::MAX_MOVE_SLOTS;
            let mut policy = [0.0f32; ACTION_DIM];
            for p in policy.iter_mut().take(n) {
                *p = 1.0 / n as f32;
            }
            OracleOut {
                policy,
                value: self.value,
            }
        }

        async fn ack_empty(&self, _player: Player) {}
    }

    fn is_normalized(p: &[f32; ACTION_DIM]) {
        let total: f32 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "not normalized: {p:?}");
        for v in p {
            assert!(*v >= 0.0, "negative prob: {p:?}");
        }
    }

    #[tokio::test]
    async fn lookahead_terminates_and_returns_sane_targets() {
        let start = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::GoodraHisui);
        let cfg = LookaheadConfig {
            sims: 32,
            randomize: false,
            crit_enabled: false,
            ..LookaheadConfig::default()
        };
        let oracle = UniformOracle { value: 0.5 };
        let r = run_lookahead(&oracle, start, Player::P1, 12345, &cfg).await;
        is_normalized(&r.training_pi);
        is_normalized(&r.selection_pi);
        assert!(
            (0.0..=1.0).contains(&r.value),
            "value out of range: {}",
            r.value
        );
    }

    #[tokio::test]
    async fn lookahead_sharpens_toward_higher_win_rate_move() {
        // 決定論モードで両側 Cloyster。policy は一様なので、勝率の高い手 (= P1 視点で
        // 先に相手を倒せる手) に training_pi が偏るはず。少なくとも一様 (0.5) からは
        // 動くことを確認する。
        let start = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::GoodraHisui);
        let cfg = LookaheadConfig {
            sims: 64,
            randomize: false,
            crit_enabled: false,
            ..LookaheadConfig::default()
        };
        let oracle = UniformOracle { value: 0.5 };
        let r = run_lookahead(&oracle, start, Player::P1, 999, &cfg).await;
        let spread = (r.training_pi[0] - r.training_pi[1]).abs();
        assert!(spread > 1e-3, "training_pi did not move from uniform: {:?}", r.training_pi);
    }

    #[tokio::test]
    async fn parallel_rollouts_produce_sane_targets() {
        // sim_concurrency > 1 (スライディングウィンドウ) でも正常終了し、各 arm が最低 1 回
        // 評価され、value/π が健全な範囲に収まることを確認する。
        let start = BattleState::new(Stage::Stage3a, SpeciesId::Cloyster, SpeciesId::GoodraHisui);
        let cfg = LookaheadConfig {
            sims: 64,
            sim_concurrency: 8,
            randomize: false,
            crit_enabled: false,
            ..LookaheadConfig::default()
        };
        let oracle = UniformOracle { value: 0.5 };
        let r = run_lookahead(&oracle, start, Player::P1, 4242, &cfg).await;
        is_normalized(&r.training_pi);
        is_normalized(&r.selection_pi);
        assert!((0.0..=1.0).contains(&r.value), "value out of range: {}", r.value);
        let spread = (r.training_pi[0] - r.training_pi[1]).abs();
        assert!(spread > 1e-3, "training_pi did not move from uniform: {:?}", r.training_pi);
    }

    #[tokio::test]
    async fn stage3b_unfavorable_matchup_favors_switch() {
        use crate::observation::{ACTION_DIM, MAX_MOVE_SLOTS};
        use poke_sho_rust::scenario::TeamId;
        // P1 = Team2 先発 Cloyster (Bulldoze: 対 Cloyster は非 SE)。控えは Team2 Goodra
        // (Shock Wave: 対 Cloyster に SE)。相手 P2 = Team1 先発 Cloyster (Water)。
        // よって P1 は「攻撃」より「控え Goodra へ交代」が有利。交代手 (index MAX_MOVE_SLOTS)
        // の training_pi が攻撃手より高く立つことを期待する。
        let start = BattleState::new_with_teams(
            Stage::Stage3b,
            (TeamId::Team2, 0),
            (TeamId::Team1, 0),
        );
        // 深さ上限が浅いと rollout が終局せず value=0.5 で縮退し、手の差が出ない。
        // 終局まで読める深さを与えて勝率差を観測する。
        // 攻撃を撃ち合う方策の下で rollout を評価する。一様方策だと交代のテンポ損が SE
        // 利得を打ち消し勝率が拮抗するが、攻撃優先方策なら本来の傾向が決定的に出る:
        // 居座る (Bulldoze 等倍) と相手 Shock Wave の SE で競り負け、控え Goodra へ交代
        // すると相手の攻撃が等倍化し SE を返せる。
        let cfg = LookaheadConfig {
            sims: 256,
            search_turn_min: 12,
            search_turn_max: 16,
            randomize: false,
            crit_enabled: false,
            ..LookaheadConfig::default()
        };
        let oracle = AttackGreedyOracle { value: 0.5 };
        assert_eq!(ACTION_DIM, 9); // MAX_MOVE_SLOTS(4) + 交代枠(MAX_PARTY-1=5)。
        let r = run_lookahead(&oracle, start, Player::P1, 7, &cfg).await;
        is_normalized(&r.training_pi);
        // 各候補手の rollout 平均勝率で判定する。training_pi は prior (AttackGreedyOracle が
        // 交代に 0 を与える) に引きずられるため、ここでは純粋な手の良し悪し = win_rates を見る。
        // root の合法手は Bulldoze (唯一の習得技 = スロット 0) と 交代 (=index MAX_MOVE_SLOTS)。
        let switch_wr = r.win_rates[MAX_MOVE_SLOTS];
        let attack_wr = r.win_rates[0];
        assert!(
            switch_wr > attack_wr,
            "switch should win more in unfavorable matchup: switch_wr={switch_wr} attack_wr={attack_wr} win_rates={:?}",
            r.win_rates
        );
    }

    #[test]
    fn depth_skew_biases_toward_deeper_caps() {
        // skew=1.0 は一様、skew=2.0 は深側に偏ることを実分布で確認する。
        let mut rng = BattleChaCha::from_u64(123, false, false);
        let base = LookaheadConfig {
            search_turn_min: 4,
            search_turn_max: 8,
            ..LookaheadConfig::default()
        };
        let mean = |skew: f32| {
            let cfg = LookaheadConfig { depth_skew: skew, ..base };
            let mut r = BattleChaCha::from_u64(123, false, false);
            let n = 20_000u32;
            let mut sum = 0u64;
            for _ in 0..n {
                let d = sample_depth_cap(&cfg, &mut r);
                assert!((4..=8).contains(&d));
                sum += d as u64;
            }
            sum as f64 / n as f64
        };
        let _ = &mut rng;
        let uniform_mean = mean(1.0);
        let skewed_mean = mean(2.0);
        // 一様の理論平均は 6.0 付近。skew>1 はそれより深い側へ寄る。
        assert!((uniform_mean - 6.0).abs() < 0.1, "uniform_mean={uniform_mean}");
        assert!(skewed_mean > uniform_mean + 0.5, "skewed_mean={skewed_mean} uniform_mean={uniform_mean}");
    }
}
