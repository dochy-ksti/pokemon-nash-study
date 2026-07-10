---
name: publish-web
description: 静的Web対戦アプリ (ポケモンにおけるナッシュ均衡の研究) を再ビルドして公開する。変更内容に応じて wasm 再生成・ポリシーテーブル再計算を行い、整合検証してから commit/push で反映する。「Webアプリを公開/更新して」「web を再ビルドしてデプロイ」等で使う。
user-invocable: true
argument-hint: [--stage 3b|3c] [--rebuild web|wasm|table|all] ["コミットメッセージ"]
allowed-tools: [Bash, Read, Edit, Grep, Glob]
---

# publish-web

学習済み policy AI と GPU 不要で対戦できる静的Webアプリ (`web/`) を、
変更に応じて再ビルドし、整合を確認してから公開するスキル。

> **配信中の 3b AI は PSRO_nash3 の σ 混合 (メタ Nash・15 体/カバー 98.1%)**。単一ネットではなく、
> PSRO 集団 Π の Nash 混合 σ を各状態で σ 加重平均した 1 枚テーブル。生成は
> `scripts/export_policy_table_mixture.py` (手順 3)。過去の単一ネット版 (K1b5_ep280.pt +
> `export_policy_table.py`) は下位互換で残るが、既定は混合版。
> **勝率表示 (value 列) だけは自己対戦 warmup の `PSRO_nash3_c0` から焼く** (`--value-checkpoint`)。
> BR 群の value は互角局面を過大評価するため (手順 3 参照)。meta の `value_source` が出所を記録。

- **このリポジトリが開発元かつ公開先を兼ねる** (github.com/dochy-ksti/pokemon-nash-study)。
  wasm もポリシーテーブルもここでビルドし、`web/` をそのまま配信する。別リポジトリへの同期は無い。
- Cloudflare Pages が Git 連携で自動デプロイ (Build output directory = `web`)。
- 反映は `scripts/deploy-web.sh` が commit→push を1コマンドで行う (in-place)。

## 前提・設計 (変更してはいけない不変条件)

- **クロスチーム限定**: 学習は必ず Team1 vs Team2 のみ (game_task.rs)。同チーム対戦は未学習=OOD。
  テーブルは `opp_team = 1 - ai_team` を次元に持たない (radix 7 次元、総数 8·H^4)。app.js の
  `denseIndex` と meta.json の radix、列挙器 (policy_table.rs) はこの前提で一致していること。
- **配信バトル規則**: 乱数あり・急所あり・命中100%。ターン解決はブラウザ内 WASM (poke-sho-rust)。
- **HP 4% 離散化 (26 バケット)**。埋め込みは素のポリシー。配信 3b AI = PSRO_nash3 の σ 混合
  (`data/poke-ai3/tournament/PSRO_nash3_psro.json` の Π + σ)。ローカル学習資産 (`data/`,
  checkpoint, PSRO 結果 JSON, 中心スナップショット群) は gitignore 対象で再生成前提。手元に
  無ければ先に PSRO (ckpt_tournament.py psro) を回して用意する。単一ネット版なら任意の checkpoint。
- push は**ユーザーの認証**で行う。Claude は push を代行せず、`deploy-web.sh` の実行はユーザーに促す
  (または実行してよいか確認する)。

## 何を再ビルドすべきか (変更箇所からの判断)

| 変更した箇所 | 必要な再ビルド |
|---|---|
| `web/app.js` `web/index.html` `web/style.css` のみ | なし → 5. deploy だけ |
| `poke-wasm/` または `poke-sho-rust/` (sim/damage/turn/types 等) | 2. wasm 再生成 (+ 観測/モデルに波及するなら 3 も) |
| `poke-ai3-python/src/policy_table.rs`・モデル・観測 (`poke-env-rust`)・配信AI(σ混合/checkpoint)・ステージ | 3. テーブル再計算 |
| 引数 `--rebuild all` 指定時 | 2 と 3 の両方 |

`--rebuild` 未指定なら、直近の変更 (git status / 会話) から上表で判断する。迷ったら `all`。

## 手順

### 1. ネイティブモジュールの鮮度 (Rust を触った場合)
```bash
cd poke-ai3-python && make build   # maturin develop --release で _native.so を最新化
```
`make` 系ターゲットは実行前に必ず maturin ビルドを挟むので、後続の export/検証が古い .so で走らない。

### 2. WASM 再生成 (poke-wasm / poke-sho-rust を変えた場合)
```bash
wasm-pack build poke-wasm --target web --release --out-dir ../web/pkg   # リポジトリ root から
```
生成物 `web/pkg/poke_wasm.js` `poke_wasm_bg.wasm` が更新される (約84KB)。
初回のみ `rustup target add wasm32-unknown-unknown` と `cargo install wasm-pack` が要る (ネット取得。実行前に確認)。

### 3. ポリシーテーブル再計算 (モデル/観測/enumerator/配信AI/ステージを変えた場合)
GPU が要る。実行前にコマンドと引数をユーザーに説明すること。出力は `web/policy_{stage}.bin`・
`web/value_{stage}.bin` (同じ列挙順・長さの u16 = round(value*1000)、勝率テーブル)・
`web/policy_{stage}.meta.json` の 3 点。3b は valid ≈ 3,380,000 / total 3,655,808、各 bin ≈ 7.3MB。

**(既定) σ 混合版** — PSRO 結果の Π + σ から σ 加重平均テーブルを焼く。各状態で全サポートネットを
推論するのでコストはサポート数に比例 (単一ネットの N 倍)。裾は `--sigma-floor` で刈って再正規化
(既定 0.005)。15 体規模で ~30〜60 分なので **バックグラウンド実行推奨** (setsid nohup + ログ poll)。
実行前に旧 `web/policy_3b.bin`/`value_3b.bin`/`meta.json` をバックアップしておく。
```bash
cd poke-ai3-python && uv run python scripts/export_policy_table_mixture.py \
  --psro-json ../data/poke-ai3/tournament/PSRO_nash3_psro.json \
  --sigma-floor 0.005 --stage 3b --hp-buckets 26 --out-dir ../web \
  --value-checkpoint ../data/poke-ai3/tournament/PSRO_nash3_c0.pt
```
meta には `value_scale`・`mixture` (psro_json / sigma_floor / cover / members[ckpt,sigma,weight])・
`value_source` が入る。手順 4 の検証は `mixture.members` を σ 加重平均で (policy)、`value_source`
の単一ネットで (value) 突き合わせる。

**`--value-checkpoint` (勝率表示の較正・強く推奨)**: 着手 (policy) はσ混合のまま、勝率表示
(value 列) だけこの単一 checkpoint から焼く。**BR/exploiter 群の value head は「倒すべき相手
(σ混合) 前提」で互角局面を過大評価する** (実測: 対称開始で各側 ~0.68・your+foe 和 ~1.36。本来は
和 1.0)。自己対戦 warmup ネット (PSRO の `*_c0.pt` = Π 空 warmup で自己対戦のみ) を指定すると
ミラー基準の自然な勝率になる (対称開始 ~0.574・value mean 0.61→0.53)。過大評価は互角付近に集中し、
明確な有利/不利局面は全ネット ~1.0 / ~0.0 で一致する。省略すると value もσ混合 (過大評価が残る)。

**(下位互換) 単一ネット版** — 任意の 1 checkpoint をそのまま焼く。5 分未満。
```bash
cd poke-ai3-python && make export-policy-table ARGS="--checkpoint <path.pt> \
  --stage 3b --hp-buckets 26 --out-dir ../web"
```
注意: 現行の `export_policy_table.py` は policy のみ出力し value・value_scale を書かない
(配信には value_3b.bin が要る)。単一ネット配信に戻すなら value も出す形へ要拡張。

3c を出す場合は 3c 個体を指定し `--stage 3c` に (別途 funnel で選抜が必要。app.js のステージ対応も
要一般化)。

### 4. 整合検証 (テーブルを作り直したら必須)
「JS の denseIndex = Rust の列挙 index」かつ「テーブル値 = 配信AIを直接 infer した P(交代)/value」が
一致することを、有効 index の無作為サンプルでスポット検証する。**policy はσ混合 (`mixture.members`
の σ 加重平均)、value は `value_source` の単一ネット** (無ければσ混合と同一) で突き合わせる。
`poke-ai3-python` から:
```bash
uv run python - <<'PY'
import numpy as np, json
from pathlib import Path
from poke_ai3 import enumerate_policy_batch, MAX_MOVE_SLOTS
from poke_ai3_train.agent import Agent
meta=json.load(open('../web/policy_3b.meta.json')); H=meta['hp_buckets']; PROB=meta['prob_scale']; SENT=meta['sentinel']; VSCALE=meta['value_scale']
PUB=Path('../data/poke-ai3/tournament')
tab=np.fromfile('../web/policy_3b.bin',dtype=np.uint16)
vtab=np.fromfile('../web/value_3b.bin',dtype=np.uint16)
# policy 構成: σ混合なら mixture.members、単一ネットなら checkpoint 1 体を weight 1.0 とみなす
mix=meta.get('mixture',{}).get('members') or [{'ckpt':meta['checkpoint'],'weight':1.0}]
# value 構成: value_source があればその単一ネット、無ければ policy と同一混合
vsrc=meta.get('value_source')
# JS denseIndex mirror (cross-team, radix: ai_team,ai_active,ai_hp_c,ai_hp_g,opp_active,opp_hp_c,opp_hp_g)
def jsindex(at,aa,ac,ag,oa,oc,og):
    k=at;k=k*2+aa;k=k*H+ac;k=k*H+ag;k=k*2+oa;k=k*H+oc;k=k*H+og;return k
pagents=[(Agent(device='cuda',checkpoint_path=PUB/(m['ckpt']+'.pt'),infer_graph=False),m['weight']) for m in mix]
vagents=([(Agent(device='cuda',checkpoint_path=PUB/(vsrc+'.pt'),infer_graph=False),1.0)] if vsrc else pagents)
ok=True
for k in np.random.default_rng(0).choice(np.where(tab!=SENT)[0],6,replace=False):
    r=int(k); og=r%H;r//=H; oc=r%H;r//=H; oa=r%2;r//=2; ag=r%H;r//=H; ac=r%H;r//=H; aa=r%2;r//=2; at=r%2
    obs,idx=enumerate_policy_batch('3b',H,int(k),1); assert idx[0]==k
    p=v=0.0
    for ag_,w in pagents:
        pol,_=ag_.infer_encoded(obs); p+=w*float(pol[:,MAX_MOVE_SLOTS:].sum(1)[0])
    for ag_,w in vagents:
        _,val=ag_.infer_encoded(obs); v+=w*float(val[0])
    # value head は [0,1] 外に僅かに出る (負けきり局面で負値等)。テーブルは clip 済みなので比較も clip する。
    exp_p=min(max(round(p*PROB),0),PROB); exp_v=min(max(round(v*VSCALE),0),VSCALE)
    m=(jsindex(at,aa,ac,ag,oa,oc,og)==k) and abs(exp_p-int(tab[k]))<=1 and abs(exp_v-int(vtab[k]))<=1; ok&=m
    print(k,'idx_ok',jsindex(at,aa,ac,ag,oa,oc,og)==k,'table',int(tab[k]),'direct',exp_p,'value',int(vtab[k]),exp_v)
print('ALL OK' if ok else 'FAIL')
PY
```
`ALL OK` を確認する。あわせて `total == len(tab)`、`node --check web/app.js` も見ておく。

### 5. 公開 (deploy)
```bash
scripts/deploy-web.sh "変更内容の短い説明"
```
web/ (+ README/LICENSE) を commit → push → Cloudflare が自動再デプロイ。変更が無ければ何もしない。
push はユーザー認証で走るため、**このコマンドの実行はユーザーに促すか、実行可否を確認する**。

## 確認リンク
- トップ = 研究概要ランディング (`web/index.html`)。対戦アプリは `web/battle-3b.html` に分離。
  両ページとも `?lang=` (既定 en / `?lang=ja`) で日英切替し、リンク遷移時も言語を引き継ぐ。
- 研究概要 (英語): `https://<repo>.pages.dev/` / 日本語: `.../?lang=ja`
- 3b 対戦 (英語): `https://<repo>.pages.dev/battle-3b.html` / 日本語: `.../battle-3b.html?lang=ja`
