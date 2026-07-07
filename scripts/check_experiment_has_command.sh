#!/usr/bin/env bash
# PostToolUse フック: experiments/poke-ai3/ 配下の実験記録に「実行コマンド」の
# コードブロックが含まれているかを検査する。AGENTS.md の Experiments 規約
# 「Record experiment purpose, command, and result」を機械的に強制するためのもの。
#
# stdin に Claude Code のフック入力 JSON を受け取り、対象ファイルにコマンド
# ブロックが無ければ decision=block を返して Claude に追記を促す。
set -euo pipefail

input="$(cat)"

file="$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_response.filePath // empty')"

# 対象は experiments/poke-ai3/ 配下の .md のみ。それ以外は素通り。
case "$file" in
  */experiments/poke-ai3/*.md) : ;;
  *) exit 0 ;;
esac

# 書き込み直後のファイルが実在しなければ何もしない。
[ -f "$file" ] || exit 0

# ```bash / ```sh / ```console のいずれかのコードフェンスがあればコマンド記載とみなす。
if grep -qE '^```(bash|sh|console)[[:space:]]*$' "$file"; then
  exit 0
fi

reason=$(cat <<'MSG'
この実験記録には実行コマンドのコードブロックがありません。AGENTS.md の Experiments 規約
「Record experiment purpose, command, and result」により、実験ファイルには目的・コマンド・結果を
記録する必要があります。実際に流したコマンド（引数を省略せずそのまま）を ```bash フェンスで
追記してください。例:

```bash
cd poke-ai3-python && make train ARGS="--num-games 32 --sim-concurrency 16 ..."
```
MSG
)

jq -n --arg r "$reason" '{decision:"block", reason:$r}'
exit 0
