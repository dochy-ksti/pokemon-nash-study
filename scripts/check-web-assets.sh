#!/usr/bin/env bash
# web/ が Cloudflare Pages の制約を満たすか検査する。違反があれば exit 1。
#
# なぜ必要か: Pages は上限を超えるアセットが1つでもあるとデプロイ「全体」を失敗させる。
# しかも Git 連携では失敗が静かで、サイトは最後に成功したビルドのまま残り続ける。
# 実際に 2026-07 に policy_3d.bin (27.89MiB) がこの上限を踏み、サイトが3c時代のまま
# 5コミット分凍結していたのに数日気づかなかった。
#
# 使い方:
#   scripts/check-web-assets.sh          # web/ を検査
#   scripts/deploy-web.sh                # 内部で自動的に呼ばれる
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB="$ROOT/web"

# Cloudflare Pages の制限 (https://developers.cloudflare.com/pages/platform/limits/)
MAX_FILE_BYTES=$((25 * 1024 * 1024))   # 1ファイル 25 MiB
MAX_FILES=20000                        # 1デプロイ 20,000 ファイル

[[ -d "$WEB" ]] || { echo "error: web/ not found: $WEB" >&2; exit 1; }

fail=0

# 1. ファイルサイズ上限
oversized="$(find "$WEB" -type f -size +"${MAX_FILE_BYTES}"c -printf '%s\t%p\n' | sort -rn)"
if [[ -n "$oversized" ]]; then
  echo "NG: Cloudflare Pages の 25MiB/ファイル 上限を超える配信物があります。" >&2
  echo "    これがあるとデプロイ全体が失敗し、サイトは古いまま静かに凍結します。" >&2
  while IFS=$'\t' read -r bytes path; do
    printf '     %8.2f MiB  %s\n' "$(echo "scale=4;$bytes/1048576" | bc)" "${path#"$ROOT/"}" >&2
  done <<< "$oversized"
  echo "    対策: 方策テーブルなら u8 で焼く (export_nash_geo_web.py が自動でそうする)。" >&2
  fail=1
fi

# 2. ファイル数上限
count="$(find "$WEB" -type f | wc -l)"
if (( count > MAX_FILES )); then
  echo "NG: web/ のファイル数 $count が Pages の上限 $MAX_FILES を超えています。" >&2
  fail=1
fi

# 3. 各ステージの policy/value/meta が揃っていること (metaだけ更新して bin を忘れる事故防止)
for meta in "$WEB"/policy_*.meta.json; do
  [[ -e "$meta" ]] || continue
  stage="$(basename "$meta" .meta.json)"; stage="${stage#policy_}"
  for f in "policy_$stage.bin" "value_$stage.bin" "battle-$stage.html"; do
    if [[ ! -f "$WEB/$f" ]]; then
      echo "NG: policy_$stage.meta.json はあるのに web/$f がありません。" >&2
      fail=1
    fi
  done
done

if (( fail )); then
  echo "web アセット検査: 失敗" >&2
  exit 1
fi

printf 'web アセット検査: OK (%d ファイル, 最大 %.2f MiB)\n' \
  "$count" "$(echo "scale=4;$(find "$WEB" -type f -printf '%s\n' | sort -rn | head -1)/1048576" | bc)"
