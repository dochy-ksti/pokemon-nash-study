#!/usr/bin/env bash
# 公開サイトがローカルの web/ と同じものを配信しているか検証する。違えば exit 1。
#
# なぜ必要か: Cloudflare Pages の Git 連携はデプロイが失敗しても静かで、サイトは最後に
# 成功したビルドのまま残る。しかも Pages は未知のパスにフォールバックの index.html を
# 200 で返すため、`curl -o /dev/null -w %{http_code}` の疎通確認は「存在しない
# policy_3d.bin」に対しても 200 を返してしまい、まったく当てにならない。
# 2026-07 にこれで 3d 未公開に数日気づかなかった。中身を突き合わせるしかない。
#
# 使い方:
#   scripts/verify-deploy.sh                 # 既定のURLを検証
#   scripts/verify-deploy.sh https://... 　  # URL 指定
#   scripts/deploy-web.sh                    # push 後に内部から呼ばれる
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="${1:-https://pokemon-nash-study.pages.dev}"
SITE="${SITE%/}"
RETRIES="${VERIFY_DEPLOY_RETRIES:-10}"   # Pages の反映待ち (10回 × 15秒 = 最大2.5分)
SLEEP="${VERIFY_DEPLOY_SLEEP:-15}"

fetch() { curl -sS --max-time 30 "$1"; }

check_once() {
  local failed=0

  # 1. app.js が最新か。ビルドが古いままなら真っ先にここで差が出る。
  if ! diff -q <(fetch "$SITE/app.js") "$ROOT/web/app.js" >/dev/null 2>&1; then
    echo "  - app.js が配信版とローカルで異なる (ビルドが古い可能性)"
    failed=1
  fi

  # 2. 各ステージの meta が「本物のJSON」として配信されているか。
  #    フォールバックHTMLが返る = そのステージは実際には存在しない。
  for meta in "$ROOT"/web/policy_*.meta.json; do
    [[ -e "$meta" ]] || continue
    local name stage
    name="$(basename "$meta")"
    stage="${name#policy_}"; stage="${stage%.meta.json}"
    local tmp
    tmp="$(mktemp)"
    if ! fetch "$SITE/$name" > "$tmp"; then
      echo "  - $name の取得に失敗"; failed=1; rm -f "$tmp"; continue
    fi
    # JSON として意味的に比較する。バイト比較は末尾改行の有無で偽陽性を出す。
    case "$(python3 - "$tmp" "$meta" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        live = json.load(f)
except Exception:
    print("NOTJSON"); sys.exit()
with open(sys.argv[2]) as f:
    local = json.load(f)
print("SAME" if live == local else "DIFF")
PY
)" in
      NOTJSON)
        echo "  - $name がJSONでない (Pagesのフォールバックが返っている = Stage $stage 未公開)"
        failed=1 ;;
      DIFF)
        echo "  - $name の内容がローカルと異なる (配信版が古い)"
        failed=1 ;;
    esac
    rm -f "$tmp"
  done

  return $failed
}

echo "配信検証: $SITE"
for ((i = 1; i <= RETRIES; i++)); do
  out="$(check_once 2>&1)" && { echo "$out"; echo "配信検証: OK — 公開サイトはローカルの web/ と一致しています。"; exit 0; }
  if (( i < RETRIES )); then
    echo "  (試行 $i/$RETRIES — 反映待ち ${SLEEP}s)"
    sleep "$SLEEP"
  fi
done

echo "配信検証: 失敗 — 公開サイトがローカルと一致しません。" >&2
echo "$out" >&2
echo "" >&2
echo "確認事項:" >&2
echo "  1. Cloudflare Pages のデプロイ履歴が Failed になっていないか" >&2
echo "  2. scripts/check-web-assets.sh が通るか (25MiB上限違反はデプロイ全体を落とす)" >&2
echo "  3. コミットが push 済みか (git status -sb で ahead になっていないか)" >&2
exit 1
