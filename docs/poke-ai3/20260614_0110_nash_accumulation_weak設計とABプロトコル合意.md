# nash_accumulation_weak の設計・CLI 配線・A/B プロトコル（実装前合意）

本書は `nash.rs` の崖（「平均勝率の半分以下を training 0 に落とす」処理）を穏当化した
`nash_accumulation_weak` を新設し、通常版と A/B 比較するための合意仕様である。
**実装は本書では行わない。** 着手は別途指示を受けてから。

背景: `experiments/poke-ai3/stage3/20260613_2329_gauntlet_ep50-150_...md` の結論
「ep120 前後で学習が頭打ち。改善には nash 累積アルゴリズム側の変更が必要。崖あり/なしで
ep0→120 を学習し到達プラトー水準を総当たりで比較するのが筋」を受けた検証である。

---

## 1. weak 関数の数式仕様

記号: `a = nash_avg`、`lr = nash_learning_rate`、`w = win_rates[i]`、
`plus_max = (1.0 - a).max(EPS)`。

### 1.1 係数 `factor_weak(w)`

1.0 を中心とした乗法的に対称な乗数。区分線形（w について線形）。

| 区間 | factor | 端点 |
|---|---|---|
| `w <= a/2` | `1/lr`（フラットなフロア） | a/2 で `1/lr` |
| `a/2 < w <= a` | `((lr-1)*(w - a/2)/(a/2) + 1) / lr` | a/2→`1/lr`、a→`1.0` |
| `w > a` | `1 + (lr-1)*(w - a)/plus_max` | a→`1.0`、w=1→`lr` |

- 下端 `1/lr`、中央（`w=a`）`1.0`、上端 `lr` で連続（段差なし）。`lr=2.0` では下端 0.5・上端 2.0。
- 中段は `lr=2` のとき `w/a` に簡約される（検算済み）。上段の比 `(w-a)/plus_max` は
  `a <= w <= 1` で常に `[0,1]` に収まるため、factor は爆発せず `[1, lr]` に収まる。
- `lr` はもはや正規化で打ち消されず、**分布の鋭さを直接制御するノブ**になる
  （`lr=1` で factor≡1 → `training ∝ predicted` の「Nash 調整なし」ベースライン）。

### 1.2 training / selection

- `training[i] = predicted[i] * factor_weak(w)`（`predicted[i]` 乗算は維持）。
- 上側ブランチ（`w >= a`）のみ `nash_pi_limit`（0.05）で下限クランプ（現行と同じ。下側にはなし）。
  - 注記: weak は一律 `*lr` が無いぶん training の絶対スケールが現行より小さく、
    同じ `nash_pi_limit` でも相対的な効きがやや強い。実験記録に明記する。
- `selection[i] = max(training[i], nash_minimum_pi)` を**全合法手に一律適用**
  （現行の `w<=a/2` 特別扱い・`continue` は廃止。最小探索率 0.03 は維持）。
- 最後に `training` / `selection` をそれぞれ `normalize()`。

### 1.3 早期リターンのガード（ループ前）

現行 `nash_accumulation` の 4 ガードのうち **3 番だけ挙動を変える**:

| # | 条件 | 現行 | weak |
|---|---|---|---|
| 1 | `cnt == 0` | uniform | uniform（踏襲） |
| 2 | `cnt2 == 0` | uniform | uniform（踏襲） |
| 3 | `nash_avg >= 1 - EPS` | one_hot_best | **通常ループに流す**（1手潰しせず穏やかに分布） |
| 4 | `nash_avg <= EPS` | uniform | uniform（踏襲） |

- 3 を外すため `plus_max` のゼロ割対策として `let plus_max = (1.0 - nash_avg).max(EPS);` を入れる
  （`nash_avg == 1.0` ちょうどの `0/0` のみ回避。比は `[0,1]` 圏内なので factor は `[1,lr]` を保つ）。

---

## 2. CLI 配線

### 2.1 切り替え方式

`LookaheadConfig` に `nash_weak: bool`（default `false`）を追加し、PyO3 コンストラクタ
（`poke-ai3-python/src/lib.rs`）→ `train-loop` の `--nash-weak` フラグまで配線。
nash 算出側でフラグを見て `nash_accumulation` / `nash_accumulation_weak` を分岐。
A/B は同一バイナリでフラグ有無の 2 run（再ビルド不要）。

### 2.2 露出する CLI 引数（デフォルトは現挙動維持）

| 引数 | 型 | default | 配線先 |
|---|---|---|---|
| `--nash-weak` | bool フラグ | false | `LookaheadConfig.nash_weak` |
| `--nash-learning-rate` | float | 2.0 | `LookaheadConfig.nash_learning_rate`（現状 `..default()` 任せを明示露出） |
| `--learning-rate` | float | 1e-4 | `AgentConfig.learning_rate`（AdamW, `agent.py:21/53`） |

- `nash_weak` / `nash_learning_rate` は推論時パラメータで checkpoint の学習状態と独立。
  CLI 値をそのまま使えばよい（保存・復元の衝突なし）。
- `--learning-rate` は checkpoint に保存・復元される（`agent.py:155/181`）。
  **CLI 明示時は checkpoint 値を上書き**（ロード後に `optimizer.param_groups` の lr を再設定）。
  CLI 未指定時は従来どおり checkpoint 値を尊重。

---

## 3. A/B プロトコル

評価手法は `20260613_2329` 実験に準拠（`scripts/gauntlet.py` ＋ `scripts/anchor_sweep.py`、
`eval_ckpt_vs_ckpt.py`）。学習設定も同実験に合わせる: hidden128 / stage3b / no-random・no-crit /
sims64 / search 6-12 / num-games 32。

### 3.1 性格づけ（重要）

`20260613_2329` の通常 run は **seed 非統制**（`gauntlet.py` は seed を渡さず `train_loop` が
`secrets.randbits(64)` でランダム）かつ `ckpt_curve_ep50.pt` 始動。よって seed ペア比較は原理的に不可。
本 A/B は「seed 統制ペア比較」ではなく、**weak を ep0 から新規にプラトーまで学習し、その（必要なら
複数 seed の）ピークが既存の最良通常 ckpt を上回るか**という非対称比較から始める。

### 3.2 強さの測定軸

- 主: `anchor_sweep.py` の**強さ指標**（固定アンカー ep105・ep130 に両 side n=512、SE≈0.022）。
- 補: weak ピーク vs 通常ピークの**直接対戦**（両 side n=512）。「2329 ピーク」は強さ指標最大の
  **ep140（.559）を主**、全勝プラトーの **ep120・ep130 を副**アンカーとする。
- `eval-vs-rule` 勝率は記録どおり強さの代理に使わない。
- 非推移性（じゃんけん性）対策として、weak ピークは固定アンカーで同一スケール化したうえで、
  weak ピーク・通常 {ep120, ep130, ep140} の小総当たり（両 side n=512）も取る。

### 3.3 段階的・ペア化エスカレーション

- **Phase 1（最安）**: weak を seed S1 で ep0→~150 学習。既存 2329 通常 ckpt と比較。
  プリレジスト閾値で決定的なら結論。
  - ※「weak(ep0 新規) vs 2329(ep50 始動・seed 非統制)」の非対称比較。向きの当たりを安く見るだけ。
- **Phase 2（Phase 1 が微妙なら）**: 新 seed S2 で **weak と通常の両方を ep0→~150 ペア学習**。
  同一 seed・同一プロトコルで nash だけ違うゴールドスタンダードのペア比較。S2 で決定的なら結論。
- **Phase 3 以降**: まだ微妙なら seed#3 を**両条件**で 1 本追加…と 1 本ずつ増やす。
  weak だけ複数 seed にするのは不公平なので、追加 seed は必ず両条件ペアで回す。

### 3.4 プリレジスト判定閾値（各フェーズ後に適用）

- **決定的（そのフェーズで結論）**: weak ピークの強さ指標が 2329 ピーク水準（≈.559）から
  **±0.08（≒3.5SE）以上**離れ、**かつ**直接対戦勝率が **0.5 から ±0.06 以上**離れている。
  → その向き（強化／劣化）で結論。
- **微妙（次フェーズへ）**: いずれかが上記マージン内（seed ノイズ圏内）。
- 観点: 強くなっているか（ピーク水準）に加え、**強さ上昇の安定性**（gauntlet で新世代が直近世代に
  勝ち続けるか、anchor_sweep のトレンドが単調か）も併せて見る。

---

## 4. 未確定・実装時の注意

- weak のプラトー到達 epoch は通常（ep120 前後）と異なりうるので、`~150ep` は目安。
  頭打ちが見えなければ snapshot を追って延長判断。
- `nash.rs` は 300 行制限（AGENTS.md）。weak 追加でファイルが膨らむなら責務境界で分割。
- 実験記録は `experiments/poke-ai3/` に、JST タイムスタンプ命名で残す。
</content>
</invoke>
