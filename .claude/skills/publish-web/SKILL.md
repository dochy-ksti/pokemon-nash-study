---
name: publish-web
description: 静的Web対戦アプリ「ポケモンにおけるナッシュ均衡の研究」を再ビルド・検証・公開する。3b/3cの交代確率テーブルと3dの4行動完全方策、Wasm、日英ページを扱う。「Webを更新/公開して」「対戦ページをデプロイして」等で使う。
---

# publish-web

GPU不要の静的Web対戦アプリ (`web/`) を、変更範囲に応じて再生成し、配信物の整合を
検証してから公開する。このリポジトリが開発元かつ公開元であり、別リポジトリへの同期はない。
Cloudflare PagesはGit連携で`web/`を自動配信する。

## 現行の配信方式

- 3b/3c/3dとも、H=26・割引0.99の幾何打ち切りゲームを解いた厳密な定常Nash方策。
- 3b/3c: 各個体1技。`policy_{stage}.bin`は状態ごとの`P(交代)`（u16×1）。
- 3d: 各個体3技。`policy_3d.bin`は状態ごとの
  `[Crunch, Dark Pulse, coverage, Switch]`（u16×4）。
- `value_{stage}.bin`は同じNashゲームのP1勝率（u16×1）。
- u16は確率・勝率を1000倍して丸める。番兵は`0xFFFF`。
- 3b/3c/3dのNPZ原本は`data/poke-ai3/nash_geo/`（gitignore）、Web用bin/metaは`web/`。
- ブラウザ対戦は乱数・急所あり、ターン上限なし。幾何打ち切り自体は人間対戦へ入れない。

旧PSROネットや単一checkpointからのexportは現在の配信方式ではない。履歴調査を除き、
`export_policy_table_mixture.py`や`export_policy_table.py`を配信テーブル生成に使わない。

## 不変条件

- クロスチーム限定: `opp_team = 1 - ai_team`。同チーム対戦は扱わない。
- dense indexのradix順:
  `ai_team, ai_active, ai_hp_c, ai_hp_g, opp_active, opp_hp_c, opp_hp_g`。
- H=26では`total=3,655,808`、有効状態は3,380,000。
- JSの`denseIndex`、Rustの`Dims`、metaの`radix`を常に一致させる。
- 3dのcoverageは、現個体が持つShock WaveまたはBulldoze。技列は各個体で
  `[Crunch, Dark Pulse, coverage]`の順。
- ローカルNPZや実験JSONはコミットしない。`web/`のbin/meta/Wasmは配信物なのでコミットする。
- 関係のないdirty worktreeを巻き込まない。

## 再生成範囲の判断

| 変更 | 必要な作業 |
|---|---|
| `web/*.html` `web/app.js` `web/style.css`だけ | Web検証のみ |
| `poke-wasm/`または`poke-sho-rust/` | Wasm再生成＋Web検証 |
| ゲーム規則・ステージ・Nashソルバ | ネイティブbuild＋Nash再計算＋export＋Wasm |
| NPZは既存でexport形式だけ変更 | Web export＋整合検証 |
| `--rebuild all` | すべて |

未指定なら`git status`・`git diff`・会話から最小範囲を判断する。既存の正しいNPZがあるなら、
不要なH=26再計算は行わない。

## 手順

### 1. 変更と成果物を確認

```bash
git status --short
git diff --check
ls -lh data/poke-ai3/nash_geo web/policy_3*.bin web/value_3*.bin
```

上書き前に必要なら既存Web成果物を`/tmp`等へバックアップする。リポジトリ内にバックアップを
作らない。

### 2. ネイティブモジュールを更新

Rust/PyO3を変更した場合:

```bash
cd poke-ai3-python
make build
```

### 3. H=26 Nashを再計算

ゲーム規則・ステージ・ソルバを変えた場合のみ実行する。5分を超えうるため、開始前にコマンド、
全引数、想定時間・メモリをユーザーへ説明し、リポジトリの実験規約どおりバックグラウンド実行する。

```bash
cd poke-ai3-python
setsid nohup uv run python scripts/run_nash_geo_h26.py \
  --stage 3d --hp-buckets 26 > /tmp/psro/nash_geo_h26_3d.log 2>&1 &
```

`--stage`は対象に応じて3b/3c/3d。3dは約7分・最大11GBの実績がある。完了後に
exploitability、方策和、鏡像値、違法交代確率を検証する。

### 4. Web用テーブルを書き出す

```bash
cd poke-ai3-python
uv run python scripts/export_nash_geo_web.py --stage 3d
```

期待する形式:

- 3b/3c: `policy_width=1`、policy/valueとも7,311,616 bytes。
- 3d: `policy_width=4`、policy 29,246,464 bytes、value 7,311,616 bytes。
- metaに`stage`, `hp_buckets=26`, `policy_width`, `policy_actions`, `radix`,
  `total`, `species_order`, solver/exploitabilityが入る。

### 5. Wasmを再生成

`poke-wasm`またはシミュレータ/ステージを変更した場合:

```bash
wasm-pack build poke-wasm --target web --out-dir ../web/pkg
```

`web/pkg/poke_wasm.js`, `.d.ts`, `poke_wasm_bg.wasm`を配信物として残す。

### 6. テーブル整合検証

shape、meta、量子化の整合を確認する。`value` はNPZとbyte-for-byte一致するが、
`policy` は完全方策 (4行動) のみ u8 へ量子化して焼くのでNPZとは一致しない。

**Cloudflare Pages は 1ファイル 25MiB を超えるアセットがあるとデプロイ全体が失敗する。**
u16×4行動だと 27.9MiB でこれを超えるため u8 (scale 254 / 番兵 255、13.95MiB) にしている。
過去に 3d を u16 で焼いてデプロイが静かに失敗し、サイトが3c時代のまま数コミット分
凍結した事故があった。サイズ検査は exporter が assert しているが、下でも必ず確認する。

```bash
cd poke-ai3-python
uv run python - <<'PY'
import json
from pathlib import Path
import numpy as np

stage = "3e"
root = Path(".."); data = root / "data/poke-ai3/nash_geo"
meta = json.loads((root / f"web/policy_{stage}.meta.json").read_text())
npz = np.load(data / f"nash_geo_h26_{stage}.npz")
dtype = np.uint8 if meta.get("policy_dtype", "u16") == "u8" else np.uint16
policy = np.fromfile(root / f"web/policy_{stage}.bin", dtype=dtype)
value = np.fromfile(root / f"web/value_{stage}.bin", dtype=np.uint16)
assert np.array_equal(value, npz["value"])
assert policy.size == meta["total"] * meta["policy_width"]
assert value.size == meta["total"]
rows = policy.reshape(meta["total"], meta["policy_width"])
valid = rows[:, 0] != meta["sentinel"]
assert int(valid.sum()) == 3_380_000
if meta["policy_width"] == 4:
    sums = rows[valid].astype(np.int64).sum(axis=1)
    assert (sums == meta["prob_scale"]).all()
    assert rows[valid].max() < meta["sentinel"]      # 番兵と衝突しない
    ref = npz["policy"].reshape(meta["total"], 4)[valid].astype(float)
    ref /= ref.sum(axis=1, keepdims=True)
    err = np.abs(rows[valid] / meta["prob_scale"] - ref).max()
    assert err <= 1 / meta["prob_scale"], err
else:
    assert np.array_equal(policy, npz["policy"])     # u16 は無損失
print("ALL OK")
PY

# 25MiB 超の配信物が無いこと (1件でもあればデプロイ全体が失敗する)
find ../web -type f -size +25M | grep . && echo "NG: 25MiB 超のファイルがある" || echo "size OK"
```

3bのNPZは環境によって旧名`nash_geo_h26.npz`なので、3b検証時はexporterと同じfallbackを使う。

### 7. コード・Wasm・ブラウザ検証

```bash
node --check web/app.js
cargo test --workspace
```

Wasm上で対象stageを構築し、合法手と技列を確認する。さらに一時HTTPサーバ＋ヘッドレスChromiumで:

- 日英タイトルとチーム技構成
- 3b/3cは2ボタン、3dは3技＋交代の4ボタン
- 自分/相手の方策確率と勝率
- AIが方策から着手し、少なくとも1ターン進む
- binの代表局面確率と画面表示が一致

を確認する。テスト用サーバ/Chromiumは終了し、scratchは`/tmp`へ置く。

### 8. コミットと公開

ユーザーが「コミット」まで求めた場合は、対象ファイルだけをstageしてコミットする。AI帰属の
trailerは付けない。

ユーザーが「公開」「デプロイ」「push」まで明示した場合のみ:

```bash
scripts/deploy-web.sh "変更内容の短い説明"
```

このスクリプトは`web/`等をcommitしてpushし、Cloudflare Pagesの再デプロイを開始する。
`scripts/check-web-assets.sh`をcommit前に、`scripts/verify-deploy.sh`をpush後に自動で実行する。
既にコミット済みなら、意図したブランチとcommitを確認して通常の`git push`を使う。公開依頼が
なければpushしない。

### 9. 公開後の検証（必須）

**pushしただけで公開できたと判断してはいけない。**

```bash
scripts/verify-deploy.sh
```

Cloudflare PagesのGit連携はデプロイが失敗しても静かで、サイトは最後に成功したビルドのまま
残り続ける。さらにPagesは未知のパスにフォールバックの`index.html`を**200**で返すため、
HTTPステータスによる疎通確認は当てにならない（存在しない`policy_3d.bin`にも200が返る）。
中身を突き合わせる`verify-deploy.sh`だけが本当の確認手段である。

実際、2026-07に`policy_3d.bin`（27.89MiB）が25MiB上限を踏んでデプロイが静かに失敗し、
サイトが3c時代のまま5コミット分凍結していたのに数日気づかなかった。発覚したのは
配信中の`app.js`が3c時代のコミットと完全一致していたからである。

## URL

- 概要: `https://pokemon-nash-study.pages.dev/`（日本語は`?lang=ja`）
- 3b: `/battle-3b.html`
- 3c: `/battle-3c.html`
- 3d: `/battle-3d.html`
- 3e: `/battle-3e.html`

各対戦ページも`?lang=ja`に対応する。
