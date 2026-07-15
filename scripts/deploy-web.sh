#!/usr/bin/env bash
# 静的Web対戦アプリをこのリポジトリから直接公開する (in-place)。
#   1. ビルド済みの web/ (生成物 pkg/bin 含む) と README/LICENSE を commit
#   2. push (Cloudflare Pages が Git 連携で自動再デプロイ)
#
# このリポジトリが開発元かつ公開先を兼ねる。別リポジトリへの rsync 同期は行わない。
# wasm 再生成やポリシーテーブル再計算は publish-web スキルの手順で先に済ませておくこと。
#
# 使い方:
#   scripts/deploy-web.sh ["コミットメッセージ"]
#   - メッセージ省略時は JST タイムスタンプ入りの既定メッセージ
#
# 変更が無ければ何もせず終了する (空コミット・無駄 push をしない)。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSG="${1:-web 更新 $(TZ=Asia/Tokyo date +%Y-%m-%d\ %H:%M) JST}"

[[ -d "$ROOT/web" ]]   || { echo "error: web/ not found: $ROOT/web" >&2; exit 1; }
[[ -d "$ROOT/.git" ]]  || { echo "error: not a git repo: $ROOT" >&2; exit 1; }

cd "$ROOT"

# Pages の上限違反があるとデプロイ全体が静かに失敗し、サイトが古いまま凍結する。
# commit 前に必ず弾く。
"$ROOT/scripts/check-web-assets.sh"

git add web readme.md readme-ja.md LICENSE
if git diff --cached --quiet; then
  echo "変更なし — commit/push をスキップしました。"
  exit 0
fi
git commit -q -m "$MSG"
echo "committed: $MSG"
git push
echo "pushed. Cloudflare Pages の反映を待って配信内容を検証します..."

# Pages の Git 連携はデプロイ失敗が静かで、サイトは古いまま残る。push して終わりにせず、
# 公開URLが実際にローカルと同じものを返すまで確認する。
"$ROOT/scripts/verify-deploy.sh"
