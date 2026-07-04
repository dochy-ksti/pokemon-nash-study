//! Nash accumulation: 手ごとの平均勝率から policy の教師 (`training_pi`) と
//! 着手用分布 (`selection_pi`) を導出する (poke-poke オセロ算法の移植)。

use crate::lookahead::LookaheadConfig;
use crate::observation::ACTION_DIM;

const EPS: f32 = 1.0e-6;

/// 平均勝率から Nash accumulation で training_pi / selection_pi を作る。
/// `cfg.nash_weak` で穏当化版 (崖を除いた版) に切り替える。
pub(crate) fn nash_accumulation(
    predicted: [f32; ACTION_DIM],
    legal: [bool; ACTION_DIM],
    win_rates: [f32; ACTION_DIM],
    cfg: &LookaheadConfig,
) -> ([f32; ACTION_DIM], [f32; ACTION_DIM]) {
    if cfg.nash_weak {
        nash_accumulation_weak(predicted, legal, win_rates, cfg)
    } else {
        nash_accumulation_strict(predicted, legal, win_rates, cfg)
    }
}

/// nash_avg を計算する。早期確定 (合法手なし・候補なし・nash_avg≈0) のときは
/// `Err` に uniform 分布を載せて返す (strict/weak 共通ガード)。one_hot 化 (nash_avg≈1)
/// は strict 専用なのでここには含めない。
fn compute_nash_avg(
    legal: &[bool; ACTION_DIM],
    win_rates: &[f32; ACTION_DIM],
) -> Result<f32, [f32; ACTION_DIM]> {
    let (sum, cnt) = fold_legal(legal, win_rates, |_| true);
    if cnt == 0 {
        return Err(uniform_legal(legal));
    }
    let first_avg = sum / cnt as f32;
    let (sum2, cnt2) = fold_legal(legal, win_rates, |w| first_avg / 2.0 <= w);
    if cnt2 == 0 {
        return Err(uniform_legal(legal));
    }
    let nash_avg = sum2 / cnt2 as f32;
    if nash_avg <= EPS {
        return Err(uniform_legal(legal));
    }
    Ok(nash_avg)
}

/// 旧 (崖あり) 版。平均の半分以下の手は training を 0 に落とす。
/// 2 seed の A/B で穏当化版 (weak) と同等以下だったため非採用となり、デフォルトは weak。
/// しばらく様子見で残すが、採用機会が無ければこの関数ごと削除予定。
fn nash_accumulation_strict(
    predicted: [f32; ACTION_DIM],
    legal: [bool; ACTION_DIM],
    win_rates: [f32; ACTION_DIM],
    cfg: &LookaheadConfig,
) -> ([f32; ACTION_DIM], [f32; ACTION_DIM]) {
    let nash_avg = match compute_nash_avg(&legal, &win_rates) {
        Ok(a) => a,
        Err(u) => return (u, u),
    };
    if nash_avg >= 1.0 - EPS {
        let best = one_hot_best(&legal, &win_rates);
        return (best, best);
    }

    let plus_max = 1.0 - nash_avg;
    let minus_max = 0.0 - nash_avg;
    let mut training = [0.0f32; ACTION_DIM];
    let mut selection = [0.0f32; ACTION_DIM];
    for i in 0..ACTION_DIM {
        if !legal[i] {
            continue;
        }
        let w = win_rates[i];
        if w <= nash_avg / 2.0 {
            // 平均の半分以下の手は training 0 のまま、selection は最低値。
            selection[i] = cfg.nash_minimum_pi;
            continue;
        }
        let diff = w - nash_avg;
        if diff >= 0.0 {
            training[i] = predicted[i] * (1.0 + diff / plus_max) * cfg.nash_learning_rate;
            if training[i] < cfg.nash_pi_limit {
                training[i] = cfg.nash_pi_limit;
            }
        } else {
            training[i] = predicted[i] * (1.0 - diff / minus_max) * cfg.nash_learning_rate;
        }
        selection[i] = if training[i] < cfg.nash_minimum_pi {
            cfg.nash_minimum_pi
        } else {
            training[i]
        };
    }
    (normalize(training), normalize(selection))
}

/// 穏当化 (weak) 版。崖を除き、係数を 1.0 中心の乗数
/// (下端 `1/lr`・中央 `1.0`・上端 `lr`) に圧縮する。`lr = nash_learning_rate`。
/// - `w <= nash_avg/2`            : `1/lr` (フラットなフロア、0 に落とさない)
/// - `nash_avg/2 < w <= nash_avg` : `1/lr` → `1.0` に線形
/// - `w > nash_avg`               : `1.0` → `lr` に線形
/// strict と違い nash_avg≈1 でも one_hot 化せず通常分布する (`plus_max` は EPS 下限ガード)。
fn nash_accumulation_weak(
    predicted: [f32; ACTION_DIM],
    legal: [bool; ACTION_DIM],
    win_rates: [f32; ACTION_DIM],
    cfg: &LookaheadConfig,
) -> ([f32; ACTION_DIM], [f32; ACTION_DIM]) {
    let nash_avg = match compute_nash_avg(&legal, &win_rates) {
        Ok(a) => a,
        Err(u) => return (u, u),
    };

    let lr = cfg.nash_learning_rate;
    let half = nash_avg / 2.0;
    let plus_max = (1.0 - nash_avg).max(EPS);
    let mut training = [0.0f32; ACTION_DIM];
    let mut selection = [0.0f32; ACTION_DIM];
    for i in 0..ACTION_DIM {
        if !legal[i] {
            continue;
        }
        let w = win_rates[i];
        let factor = if w <= half {
            1.0 / lr
        } else if w <= nash_avg {
            ((lr - 1.0) * (w - half) / half + 1.0) / lr
        } else {
            1.0 + (lr - 1.0) * (w - nash_avg) / plus_max
        };
        training[i] = predicted[i] * factor;
        // 上側ブランチ (w >= nash_avg) のみ床上げ (strict と同じ)。
        if w >= nash_avg && training[i] < cfg.nash_pi_limit {
            training[i] = cfg.nash_pi_limit;
        }
        selection[i] = if training[i] < cfg.nash_minimum_pi {
            cfg.nash_minimum_pi
        } else {
            training[i]
        };
    }
    (normalize(training), normalize(selection))
}

fn fold_legal(
    legal: &[bool; ACTION_DIM],
    win_rates: &[f32; ACTION_DIM],
    pred: impl Fn(f32) -> bool,
) -> (f32, u32) {
    let mut sum = 0.0;
    let mut cnt = 0;
    for i in 0..ACTION_DIM {
        if legal[i] && pred(win_rates[i]) {
            sum += win_rates[i];
            cnt += 1;
        }
    }
    (sum, cnt)
}

fn uniform_legal(legal: &[bool; ACTION_DIM]) -> [f32; ACTION_DIM] {
    let mut p = [0.0f32; ACTION_DIM];
    for i in 0..ACTION_DIM {
        if legal[i] {
            p[i] = 1.0;
        }
    }
    normalize(p)
}

fn one_hot_best(legal: &[bool; ACTION_DIM], win_rates: &[f32; ACTION_DIM]) -> [f32; ACTION_DIM] {
    let mut best = f32::MIN;
    for i in 0..ACTION_DIM {
        if legal[i] && win_rates[i] > best {
            best = win_rates[i];
        }
    }
    let mut p = [0.0f32; ACTION_DIM];
    for i in 0..ACTION_DIM {
        if legal[i] && best - win_rates[i] <= EPS {
            p[i] = 1.0;
        }
    }
    normalize(p)
}

fn normalize(mut p: [f32; ACTION_DIM]) -> [f32; ACTION_DIM] {
    let total: f32 = p.iter().sum();
    if total <= 0.0 {
        return p;
    }
    for v in &mut p {
        *v /= total;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LookaheadConfig {
        LookaheadConfig::default()
    }

    /// 4 手: 勝率 0.1 (< nash_avg/2), 0.4, 0.6, 0.9。残りは非合法。
    fn setup() -> ([f32; ACTION_DIM], [bool; ACTION_DIM], [f32; ACTION_DIM]) {
        let mut predicted = [0.0f32; ACTION_DIM];
        let mut legal = [false; ACTION_DIM];
        let mut win_rates = [0.0f32; ACTION_DIM];
        for (i, w) in [0.1f32, 0.4, 0.6, 0.9].into_iter().enumerate() {
            predicted[i] = 0.25;
            legal[i] = true;
            win_rates[i] = w;
        }
        (predicted, legal, win_rates)
    }

    #[test]
    fn weak_removes_cliff_strict_keeps_it() {
        let (predicted, legal, win_rates) = setup();
        let mut c = cfg();

        c.nash_weak = false;
        let (strict_train, _) = nash_accumulation(predicted, legal, win_rates, &c);
        // strict は平均の半分以下 (勝率 0.1) を 0 に落とす。
        assert!(strict_train[0] <= EPS, "strict train[0]={}", strict_train[0]);

        c.nash_weak = true;
        let (weak_train, weak_sel) = nash_accumulation(predicted, legal, win_rates, &c);
        // weak は 0 に落とさない。全合法手が正。
        for i in 0..4 {
            assert!(weak_train[i] > EPS, "weak train[{i}]={}", weak_train[i]);
            assert!(weak_sel[i] > EPS, "weak sel[{i}]={}", weak_sel[i]);
        }
        // training は勝率について単調非減少。
        for i in 0..3 {
            assert!(
                weak_train[i] <= weak_train[i + 1] + EPS,
                "monotonic violated at {i}: {} > {}",
                weak_train[i],
                weak_train[i + 1]
            );
        }
        // 正規化済み (合計 1)。
        let sum: f32 = weak_train.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
    }

    #[test]
    fn weak_factor_endpoints_lr2() {
        // lr=2 では中段係数 = w/nash_avg。下端 nash_avg/2 で 1/lr=0.5、中央 nash_avg で 1.0。
        // 予測一様・正規化前の比だけ確認するため、勝率 nash_avg/2 と nash_avg の 2 手を置く。
        let mut predicted = [0.0f32; ACTION_DIM];
        let mut legal = [false; ACTION_DIM];
        let mut win_rates = [0.0f32; ACTION_DIM];
        // nash_avg は候補手の平均。勝率を 0.5 と 1.0 にすると first_avg=0.75,
        // 候補 (>=0.375) は両方残り nash_avg=0.75。
        for (i, w) in [0.5f32, 1.0].into_iter().enumerate() {
            predicted[i] = 0.5;
            legal[i] = true;
            win_rates[i] = w;
        }
        let mut c = cfg();
        c.nash_weak = true;
        // nash_avg=0.75 のとき手0の勝率0.5 は nash_avg/2=0.375 と nash_avg=0.75 の中段。
        // factor0 = 0.5/0.75, 手1 は上端 (w=1.0 > nash_avg) factor1 = 1 + (w-na)/plus_max。
        let (train, _) = nash_accumulation(predicted, legal, win_rates, &c);
        // 上端手が中段手より大きいこと (穏当でも順序は保つ)。
        assert!(train[1] > train[0]);
    }
}
