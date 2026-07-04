# 敵先読みの効果 — --enemy-lookahead vs policy-only (K=1)

実行完了: 2026-07-03 (JST) / K=1・r=0.5・block=5・敵先読み ON (K1b5L) vs 敵 policy-only (K1b5)

## 目的・仮説

敵混合の敵 P2 は従来 policy-only (探索なし単発推論)。敵も学習者 P1 と同じ探索設定で
着手させたら学習が良くなるか? 仮説: 強い敵の方が良い勾配を与え学習が進む。

## 実装 (commit b2150fd)

role テーブル値を 0=自己対戦 / 1=敵policy-only / 2=敵先読み に拡張。--enemy-lookahead で
role==2 を書き、敵 P2 も P1 と同じ search-turn/sims/depth-skew で着手。executor signature 不変。
効果測定ドライバ: [run_enemy_lookahead.sh](../../poke-ai3-python/scripts/run_enemy_lookahead.sh)。

## 結果: 手法平均レート (K1b5+K1b5L の全6 finalist 1プール総当たり, random+crit, n-per-side 512)

| 手法 | 敵の着手 | 手法平均 | 個体レート |
|---|---|---:|---|
| **K1b5** | policy-only (従来) | **+2.9** | −1.3 / +7.9 / +2.2 |
| K1b5L | 先読みあり | −2.9 | −2.5 / −1.6 / −4.6 |

## 結論

- **敵を先読みさせても学習は良くならず、むしろやや悪化 (−5.8 Elo)**。K1b5L の3個体すべて負側で
  コヒーレントに悪い。仮説「強い敵の方が学習が進む」は支持されず。
- ただし差 5.8 は単発ラン間ノイズ帯 (±5, experiments 20260702_1312 で確認) に近く「明確に悪い」
  とは断定不可。少なくとも「先読み敵で良くなる」は否定。
- 解釈: 先読みする凍結敵は「古い方策+探索」で強いが陳腐な相手。P2 は学習除外なので P1 が
  「強い旧戦術を倒す」方向に偏り、非推移性への汎化耐性が薄れた可能性。生成コストも増える
  (敵ゲームも探索) ため費用対効果でも policy-only が優位。
- **敵は policy-only のままで良い** (--enemy-lookahead 既定 off が妥当)。確定させるなら再現ラン追加。
