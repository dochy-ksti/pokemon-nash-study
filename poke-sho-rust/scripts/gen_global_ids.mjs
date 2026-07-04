// 使い捨て生成スクリプト: pokemon-showdown の dex データから
// 種族・技・タイプのグローバル ID 表 (CSV) を生成する。
// ID は一度生成したら永久に不変。再生成は「追記」目的でのみ行い、
// 既存行の ID を変えてはならない。
//
// 実行: node poke-sho-rust/scripts/gen_global_ids.mjs
import { createRequire } from "node:module";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { Dex } = require(join(here, "../../pokemon-showdown/dist/sim"));

const outDir = join(here, "../data");

// 種族: dex ファイル順 (全国図鑑番号順・フォルムは基本形の直後)。CAP 等 num<1 は除外。
const species = Dex.species.all().filter((s) => s.num >= 1);
const speciesRows = species.map(
  (s, i) => `${i}\t${s.name}\t${s.num}\t${s.types.join("/")}`
);
writeFileSync(
  join(outDir, "species_ids.tsv"),
  "id\tname\tdex_num\ttypes\n" + speciesRows.join("\n") + "\n"
);

// 技: dex ファイル順。num<1 (CAP 専用等) は除外。Z/Max 技も ID は予約しておく。
const moves = Dex.moves.all().filter((m) => m.num >= 1);
const moveRows = moves.map(
  (m, i) => `${i}\t${m.name}\t${m.type}\t${m.category}\t${m.basePower}`
);
writeFileSync(
  join(outDir, "move_ids.tsv"),
  "id\tname\ttype\tcategory\tbase_power\n" + moveRows.join("\n") + "\n"
);

// タイプ: 18 種 + Stellar 等。dex 順。
const types = Dex.types.all();
const typeRows = types.map((t, i) => `${i}\t${t.name}`);
writeFileSync(join(outDir, "type_ids.tsv"), "id\tname\n" + typeRows.join("\n") + "\n");

console.log(
  `species=${species.length} moves=${moves.length} types=${types.length}`
);
