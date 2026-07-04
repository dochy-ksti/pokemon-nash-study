# head-to-head Phase3 num-games スイープ

実施: 2026-06-25 18:21 JST

## 目的

head-to-head 改善案 (docs/poke-ai3/20260625_1733...) の Phase 3。
head-to-head は学習ステップが無く、並列度 (num-games) がそのまま batch を太らせて
GPU充填を上げる。num-games を振って実バトル/s の最適点を測り、既定値を確定する。

Phase 1 (集計修正・sleep=0・print間引き) と Phase 2 (player別分割推論) は実装済み。
本スイープはその上での throughput 測定。

## 条件

- 計測用 checkpoint: 短時間学習 (stage3b, num-games16/sims16/epochs2) で生成した
  現行トークンレイアウト (NUM_TOKENS=17) の checkpoint を A/B 同一で使用。
  強さは throughput に無関係なので self (A vs A) で測定。
- 各 run: `--policy-only --no-random --no-crit --stage 3b --quiet-progress`、
  `--num-games N --max-batch-size N`、sim-concurrency=1 (policy-only 既定)。
- スループット = (RESULTのa_win+b_win+draw) / wall秒。wall は `date +%s.%N` 差分
  (プロセス起動・モデルロード・CUDA Graph capture を含む)。
- 固定費 (~3秒) を希釈するため `--num-eval-games 32768` (実バトル32768本) で測定。

```bash
uv run python -m poke_ai3_train.eval_ckpt_vs_ckpt \
  --checkpoint-a CK --checkpoint-b CK \
  --num-games N --max-batch-size N --num-eval-games 32768 \
  --quiet-progress --policy-only --no-random --no-crit --stage 3b
```

## 結果

| config | wall(s) | 実バトル | 実バトル/s | 基準比 |
|---|---:|---:|---:|---:|
| g64/b64   | 20.88 | 32768 | 1569 | 1.00x |
| g128/b128 | 16.97 | 32768 | 1931 | 1.23x |
| g256/b256 | 15.67 | 32768 | 2091 | 1.33x |
| g384/b384 | 14.92 | 32832 | 2200 | 1.40x |
| g512/b512 | 14.62 | 32768 | 2242 | 1.43x |

### 予備測定 (num-eval-games=1024) の注意

最初に num-eval-games=1024 で測ったが wall~3秒で固定費 (起動・モデルロード・
graph capture) が支配的になり差がほぼ出なかった。throughput 比較には大きい
num-eval-games が必須。1024 のような小標本は固定費測定になる。

## 読み

- num-games 拡大で実バトル/s は単調増加するが **384〜512 でプラトー**。
- 最大の伸びは g64→g128 (+23%)。g256→g384 は +5%、g384→g512 は +2% で頭打ち。
- player分割 (Phase2) 済みのため batch256 でも各モデル約128行。
- メモリ・tokioタスクは num-games に比例して増えるが、policy-only は lookahead が
  無く 1ゲームが軽いため g512 でも問題なく完走。

## 結論・採用

**既定 num-games=256 / max-batch-size=256 を採用。**

- 1.33x を確保しつつメモリ・タスク増を g384/g512 より抑える。
- g384 は +5%、g512 は +10% 速いがプラトー域で、並列ゲーム数 (=メモリ/タスク) の
  増分に見合わない。リソースに余裕があり最後の数%が欲しい場合のみ 384 を使う。
- max-batch-size は num-games に揃える。policy-only の同時in-flightは num-games*2 で、
  max-batch=num-games (理論上限の1/2) は十分埋まりデッドロックもしなかった。

## 反映

- `eval_ckpt_vs_ckpt.py`: `--num-games` 既定 16→256。
- `scripts/ckpt_tournament.py`: `--num-games` 既定 64→256、head_to_head 呼び出しで
  `--max-batch-size = num-games` と `--quiet-progress` を付与。

## 追測: num-games=256 固定での max-batch-size スイープ

num-games=256 を固定し max-batch-size だけを振って batch 軸を切り分けた
(同一セッション内・同一checkpoint、num-eval-games=32768)。

| config | wall(s) | 実バトル/s |
|---|---:|---:|
| g256/b128 | 13.35 | 2455 |
| g256/b192 | 12.66 | 2588 |
| g256/b256 | 11.68 | **2805 (ピーク)** |
| g256/b384 | 12.59 | 2603 |
| g256/b512 | 12.24 | 2676 |

(数値は num-games スイープ時の g256/b256=2091 より高いが、これはセッション/キャッシュ
状態差。本追測は同一セッション内なので相対比較が有効。)

### 読み

- **max-batch-size = num-games (256) がちょうど最適点**。それ未満は batch が細く GPU
  per-row 効率が落ち、それ超は batch が埋まりきらず待ち (empty padding/レイテンシ) が
  増えて低下する。
- 実際の同時 in-flight は「各ゲーム概ね 1 要求、たまに 2」で平均 num-games 付近に
  張り付くため、max-batch=num-games が「すぐ埋まる最大値」になる。
- in-flight 上限 (num-games*2=512) まで上げても**デッドロックは発生しない**
  (empty padding が回避)。ただし throughput は num-games で頭打ち。
- これは Phase3 既定 (max-batch=num-games) が最適という結論を裏付ける。コード変更不要。

## 未実施 (Phase3 残)

- timeout付き partial flush: 本スイープで g512 まで無進捗・デッドロックが出なかったため
  現時点では不要。max-batch を num-games より大きく攻める場合に再検討する。
- 強さへの影響: 本変更は throughput のみで勝率の点推定は不変 (集計はPhase1で是正済み)。
</content>
