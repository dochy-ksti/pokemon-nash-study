/* tslint:disable */
/* eslint-disable */

/**
 * 1 バトルのラッパ。`BattleState` は `Copy` なので `step` で丸ごと差し替える。
 */
export class Battle {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * `attacker` が技スロット `move_slot` を、居座り相手の active に当てた場合の
     * 通常時(急所なし) min–max ダメージと相性倍率。
     */
    damageRange(attacker: number, move_slot: number): any;
    /**
     * `player` の合法手を `[kind, arg, kind, arg, ...]` の平坦配列で返す。
     */
    legal(player: number): Uint8Array;
    /**
     * 満タンで開始する。`stage` は "3b"/"3c"、team は 0=Team1/1=Team2、active は先発 index。
     */
    constructor(stage: string, team1: number, active1: number, team2: number, active2: number);
    /**
     * 任意 HP を設定する (初期局面をカスタムしたい場合)。side 0/1・member はパーティ index。
     */
    setHp(side: number, member: number, hp: number): void;
    /**
     * 現在状態の表示ビュー (JS が描画・テーブル index 計算に使う)。
     */
    snapshot(): any;
    /**
     * 両者の手を渡して1ターン解決する。強制交代は控えが一意なので自動解決する。
     * `seed` は JS 由来 (Math.random) の乱数種。乱数・急所を有効化して本番設定を再現。
     */
    step(c1_kind: number, c1_arg: number, c2_kind: number, c2_arg: number, seed: number): any;
}

/**
 * PokeType 名の一覧 (JS 側の相性色分け等に使える補助)。未使用でも export しておく。
 */
export function typeName(idx: number): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_battle_free: (a: number, b: number) => void;
    readonly battle_damageRange: (a: number, b: number, c: number) => any;
    readonly battle_legal: (a: number, b: number) => [number, number];
    readonly battle_new: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly battle_setHp: (a: number, b: number, c: number, d: number) => void;
    readonly battle_snapshot: (a: number) => any;
    readonly battle_step: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
    readonly typeName: (a: number) => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
