---
name: publish-web
description: 静的Web対戦アプリ (ポケモンにおけるナッシュ均衡の研究) を再ビルドして公開する。変更内容に応じて wasm 再生成・ポリシーテーブル再計算を行い、整合検証してから commit/push で反映する。「Webアプリを公開/更新して」「web を再ビルドしてデプロイ」等で使う。
user-invocable: true
argument-hint: [--stage 3b|3c] [--rebuild web|wasm|table|all] ["コミットメッセージ"]
allowed-tools: [Bash, Read, Edit, Grep, Glob]
---

# publish-web

学習済み policy-only AI と GPU 不要で対戦できる静的Webアプリ (`web/`) を、
変更に応じて再ビルドし、整合を確認してから公開するスキル。

- **このリポジトリが開発元かつ公開先を兼ねる** (github.com/dochy-ksti/pokemon-nash-study)。
  wasm もポリシーテーブルもここでビルドし、`web/` をそのまま配信する。別リポジトリへの同期は無い。
- Cloudflare Pages が Git 連携で自動デプロイ (Build output directory = `web`)。
- 反映は `scripts/deploy-web.sh` が commit→push を1コマンドで行う (in-place)。

## 前提・設計 (変更してはいけない不変条件)

- **クロスチーム限定**: 学習は必ず Team1 vs Team2 のみ (game_task.rs)。同チーム対戦は未学習=OOD。
  テーブルは `opp_team = 1 - ai_team` を次元に持たない (radix 7 次元、総数 8·H^4)。app.js の
  `denseIndex` と meta.json の radix、列挙器 (policy_table.rs) はこの前提で一致していること。
- **配信バトル規則**: 乱数あり・急所あり・命中100%。ターン解決はブラウザ内 WASM (poke-sho-rust)。
- **HP 4% 離散化 (26 バケット)**。埋め込みは素のポリシー。3b 個体 = `data/poke-ai3/tournament/K1b5_ep280.pt`。
  ローカル学習資産 (`data/`, checkpoint) は gitignore 対象で再生成前提。手元に無ければ先に学習/選抜で用意する。
- push は**ユーザーの認証**で行う。Claude は push を代行せず、`deploy-web.sh` の実行はユーザーに促す
  (または実行してよいか確認する)。

## 何を再ビルドすべきか (変更箇所からの判断)

| 変更した箇所 | 必要な再ビルド |
|---|---|
| `web/app.js` `web/index.html` `web/style.css` のみ | なし → 5. deploy だけ |
| `poke-wasm/` または `poke-sho-rust/` (sim/damage/turn/types 等) | 2. wasm 再生成 (+ 観測/モデルに波及するなら 3 も) |
| `poke-ai3-python/src/policy_table.rs`・モデル・観測 (`poke-env-rust`)・checkpoint・ステージ | 3. テーブル再計算 |
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

### 3. ポリシーテーブル再計算 (モデル/観測/enumerator/checkpoint/ステージを変えた場合)
GPU + checkpoint が要る。5分未満だが、実行前にコマンドと引数をユーザーに説明すること。
```bash
cd poke-ai3-python && make export-policy-table ARGS="--checkpoint ../data/poke-ai3/tournament/K1b5_ep280.pt \
  --stage 3b --hp-buckets 26 --out-dir ../web"
```
- 3b: valid ≈ 3,380,000 / total 3,655,808、`web/policy_3b.bin` ≈ 7.3MB。
  同時に `web/value_3b.bin` (同じ列挙順・長さの u16 = round(value*1000)、勝率テーブル) も出力される。
- 3c を出す場合は 3c 個体の checkpoint を指定し `--stage 3c` に (別途 funnel で選抜が必要。app.js の
  ステージ対応も要一般化)。

### 4. 整合検証 (テーブルを作り直したら必須)
「JS の denseIndex = Rust の列挙 index」かつ「テーブル値 = 同 checkpoint を直接 infer した P(交代)」が
一致することを、有効 index の無作為サンプルでスポット検証する。`poke-ai3-python` から:
```bash
uv run python - <<'PY'
import numpy as np, json
from pathlib import Path
from poke_ai3 import enumerate_policy_batch, MAX_MOVE_SLOTS
from poke_ai3_train.agent import Agent
meta=json.load(open('../web/policy_3b.meta.json')); H=meta['hp_buckets']; PROB=meta['prob_scale']; SENT=meta['sentinel']; VSCALE=meta['value_scale']
tab=np.fromfile('../web/policy_3b.bin',dtype=np.uint16)
vtab=np.fromfile('../web/value_3b.bin',dtype=np.uint16)
# JS denseIndex mirror (cross-team, radix: ai_team,ai_active,ai_hp_c,ai_hp_g,opp_active,opp_hp_c,opp_hp_g)
def jsindex(at,aa,ac,ag,oa,oc,og):
    k=at;k=k*2+aa;k=k*H+ac;k=k*H+ag;k=k*2+oa;k=k*H+oc;k=k*H+og;return k
agent=Agent(device='cuda',checkpoint_path=Path('../data/poke-ai3/tournament/K1b5_ep280.pt'),infer_graph=False)
ok=True
for k in np.random.default_rng(0).choice(np.where(tab!=SENT)[0],6,replace=False):
    r=int(k); og=r%H;r//=H; oc=r%H;r//=H; oa=r%2;r//=2; ag=r%H;r//=H; ac=r%H;r//=H; aa=r%2;r//=2; at=r%2
    obs,idx=enumerate_policy_batch('3b',H,int(k),1); assert idx[0]==k
    pol,val=agent.infer_encoded(obs)
    p=float(pol[:,MAX_MOVE_SLOTS:].sum(1)[0]); v=float(val[0])
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
