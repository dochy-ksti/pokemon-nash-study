from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import nn

from poke_ai3 import MAX_MOVE_SLOTS

from .tables import move_gid, species_gid


@lru_cache(maxsize=1)
def _gids() -> dict[str, int]:
    """診断で使う種族・技のグローバル ID (名前→gid は Rust の ID 表が正)。
    3a/3b は {Goodra-Hisui, ShockWave, Bulldoze}、3c は {Goodra(原種), FightSpe60,
    FairyPhy60} を使う。stage を引数で受け取らず、データに現れる gid で判定するため
    両系統の gid をまとめて引く。"""
    return {
        "Cloyster": species_gid("Cloyster"),
        "GoodraHisui": species_gid("Goodra-Hisui"),
        "Goodra": species_gid("Goodra"),
        "Crunch": move_gid("Crunch"),
        "DarkPulse": move_gid("Dark Pulse"),
        "ShockWave": move_gid("Shock Wave"),
        "Bulldoze": move_gid("Bulldoze"),
        "FightSpe60": move_gid("FightSpe60"),
        "FairyPhy60": move_gid("FairyPhy60"),
    }


def _species_label(gid: int) -> str | None:
    """種族ラベル。Cloyster→Cl、Goodra-Hisui / 原種 Goodra はどちらも特殊耐久側
    として同じ役割なので "Go" に統一する (3b/3c をまたいで対面キーが揃う)。"""
    g = _gids()
    if gid == g["Cloyster"]:
        return "Cl"
    if gid in (g["GoodraHisui"], g["Goodra"]):
        return "Go"
    return None


def _se_set_for(opp_species_gid: int) -> frozenset[int]:
    """相手 active 種族に super-effective (2倍) が通る技 gid の集合 (フォルム厳密)。
    - Cloyster (Water/Ice): Shock Wave(電→水) / FightSpe60(闘→氷)。
    - Goodra-Hisui (Steel/Dragon): Bulldoze(地→鋼) / FairyPhy60(妖→竜)。
    - 原種 Goodra (Dragon): FairyPhy60(妖→竜) のみ (Bulldoze は等倍)。"""
    g = _gids()
    if opp_species_gid == g["Cloyster"]:
        return frozenset({g["ShockWave"], g["FightSpe60"]})
    if opp_species_gid == g["GoodraHisui"]:
        return frozenset({g["Bulldoze"], g["FairyPhy60"]})
    if opp_species_gid == g["Goodra"]:
        return frozenset({g["FairyPhy60"]})
    return frozenset()


def _legal_move_slots(state: dict[str, Any]) -> list[int]:
    legal_mask = state["legal_action_mask"]
    n = min(MAX_MOVE_SLOTS, len(state["my_move_gids"]), len(legal_mask))
    return [i for i in range(n) if bool(legal_mask[i])]


def _slot_of_gid(state: dict[str, Any], gid: int, require_legal: bool = True) -> int | None:
    """active の習得技スロットのうち gid の技のスロット。"""
    for slot, g in enumerate(state["my_move_gids"]):
        if int(g) != gid:
            continue
        if not require_legal:
            return slot
        legal_mask = state["legal_action_mask"]
        if slot < len(legal_mask) and bool(legal_mask[slot]):
            return slot
        return None
    return None


def _weakness_move_slot(state: dict[str, Any]) -> int | None:
    """相手種族に対する「タイプ弱点を突く技」のスロット (行動 index = 技スロット)。

    - 対 Cloyster: タイプ弱点 Shock Wave が撃てればそれ、無ければ特殊の Dark Pulse。
    - 対 Goodra-Hisui: タイプ弱点 Bulldoze が撃てればそれ、無ければ物理の Crunch。

    stage3b は混合戦略のナッシュ均衡で「正解の一手」は無い。この技は「参照となる
    弱点技」であって正解ではなく、モデルがそれを選ぶ率は行動傾向のプローブにすぎない。
    Stage3a (4技) では弱点技、Stage2a (2技) では物理/特殊の切り替えを同じ指標で測れる。"""
    g = _gids()
    opp = int(state["opp_species_gid"])
    se = _se_set_for(opp)
    if not se:
        return None
    # タイプ弱点を突く SE 技を優先し、無ければ stage3a の特殊/物理フォールバック
    # (対 Cloyster=Dark Pulse 特殊 / 対 Goodra=Crunch 物理)。
    if opp == g["Cloyster"]:
        order = [*se, g["DarkPulse"]]
    else:  # Goodra-Hisui / 原種 Goodra
        order = [*se, g["Crunch"]]
    for gid in order:
        slot = _slot_of_gid(state, gid)
        if slot is not None:
            return slot
    return None


def model_diagnostics(
    model: nn.Module,
    encoded: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, float, float]:
    """学習後のモデルの greedy 着手 (argmax) で、相手種族別に「タイプ弱点を突く技」を
    選んだ率と全体の弱点技選択率を測る。戻り値 (vs_cloyster_rate, vs_goodra_rate,
    weakness_rate)。stage3b は混合戦略のナッシュ均衡で「正解」は無いため、これは正解率
    ではなく弱点技を選ぶ行動傾向のプローブ。該当例が無い種別は -1.0。"""
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(encoded)
        choices = logits.float().argmax(dim=-1).cpu().tolist()

    vs_cloyster_total = vs_cloyster_hits = 0
    vs_goodra_total = vs_goodra_hits = 0
    hits = 0
    for item, choice in zip(examples, choices):
        state = item["state"]
        target = _weakness_move_slot(state)
        if target is None:
            continue
        hit = int(choice) == target
        # 種族ラベルで分類する (3c の原種 Goodra も "Go" に入れる)。
        label = _species_label(int(state["opp_species_gid"]))
        if label == "Cl":
            vs_cloyster_total += 1
            vs_cloyster_hits += int(hit)
        elif label == "Go":
            vs_goodra_total += 1
            vs_goodra_hits += int(hit)
        hits += int(hit)

    vs_cl = vs_cloyster_hits / vs_cloyster_total if vs_cloyster_total > 0 else -1.0
    vs_go = vs_goodra_hits / vs_goodra_total if vs_goodra_total > 0 else -1.0
    total = vs_cloyster_total + vs_goodra_total
    weakness_rate = hits / total if total > 0 else 0.0
    return vs_cl, vs_go, weakness_rate


def _active_move_gid(state: dict[str, Any]) -> int | None:
    """通常ターンで active が撃てる技の gid。Stage3b は各個体 1 技なので、
    合法な技スロットが唯一のときその gid を返す。複数/ゼロなら None。"""
    slots = _legal_move_slots(state)
    if len(slots) != 1:
        return None
    return int(state["my_move_gids"][slots[0]])


def _active_move_label(state: dict[str, Any]) -> str | None:
    """active の技ラベル。3a/3b: Shock Wave=SW / Bulldoze=BD。
    3c: FightSpe60=FS / FairyPhy60=FP。"""
    gid = _active_move_gid(state)
    g = _gids()
    if gid == g["ShockWave"]:
        return "SW"
    if gid == g["Bulldoze"]:
        return "BD"
    if gid == g["FightSpe60"]:
        return "FS"
    if gid == g["FairyPhy60"]:
        return "FP"
    return None


# 対面 (active技_v_相手種族) の列挙順と各カードの有効度メモ。3b と 3c の和集合を持ち、
# 各 run ではその stage の技だけがカウントされる (他系統の技のキーは n=0 になる)。
#   3b: SW_v_Cl=SE / SW_v_Go=半減 / BD_v_Cl=等倍 / BD_v_Go=SE
#   3c: FS_v_Cl=SE / FS_v_Go=等倍 / FP_v_Cl=等倍 / FP_v_Go=SE (対称)
SWITCH_MATCHUP_KEYS = [
    "SW_v_Cl", "SW_v_Go", "BD_v_Cl", "BD_v_Go",
    "FS_v_Cl", "FS_v_Go", "FP_v_Cl", "FP_v_Go",
]


def _switch_legal(legal_mask: list[bool]) -> bool:
    """控えへの交代スロット (MAX_MOVE_SLOTS 以降) のいずれかが合法か。"""
    return any(
        i < len(legal_mask) and bool(legal_mask[i])
        for i in range(MAX_MOVE_SLOTS, len(legal_mask))
    )


def _switch_matchup_key(state: dict[str, Any]) -> str | None:
    """(active の技 × 相手 active 種族) の対面キー。交代が合法で active が
    SW/BD のどちらかを撃てるターンだけ対象にする。"""
    opp_label = _species_label(int(state["opp_species_gid"]))
    if opp_label is None:
        return None
    legal_mask = [bool(x) for x in state["legal_action_mask"]]
    if not _switch_legal(legal_mask):
        return None
    move_label = _active_move_label(state)
    if move_label is None:
        return None
    key = f"{move_label}_v_{opp_label}"
    return key if key in SWITCH_MATCHUP_KEYS else None


def stage3b_switch_diagnostics_per_matchup(
    model: nn.Module,
    encoded: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, tuple[float, float, int]]:
    """Stage3b の交代/攻撃選択を (active の技 × 相手 active 種族) の対面ごとに測る。
    戻り値: {matchup: (model_switch_prob, teacher_switch_prob, n)} (該当ゼロは (-1,-1,0))。"""
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(encoded)
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()

    model_sum = {k: 0.0 for k in SWITCH_MATCHUP_KEYS}
    teacher_sum = {k: 0.0 for k in SWITCH_MATCHUP_KEYS}
    counts = {k: 0 for k in SWITCH_MATCHUP_KEYS}
    for item, prob in zip(examples, probs):
        key = _switch_matchup_key(item["state"])
        if key is None:
            continue
        counts[key] += 1
        model_sum[key] += sum(prob[MAX_MOVE_SLOTS:])
        target_pi = [float(x) for x in item["target_pi"]]
        teacher_sum[key] += sum(target_pi[MAX_MOVE_SLOTS:])

    return {
        k: (
            (model_sum[k] / counts[k], teacher_sum[k] / counts[k], counts[k])
            if counts[k] > 0
            else (-1.0, -1.0, 0)
        )
        for k in SWITCH_MATCHUP_KEYS
    }


def stage3b_switch_diagnostics(
    model: nn.Module,
    encoded: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, float, int]:
    """Stage3b の交代学習を、不利対面でのモデルの **交代確率** で測る。

    不利対面 = 自分の active 技が相手 active に SE を通せず、かつ生存している控えの
    技が相手 active に SE を通せる状況。戻り値 (model_switch_prob, teacher_switch_prob, n)。
    該当が無い (1v1 ステージ等) なら (-1.0, -1.0, 0)。"""
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(encoded)
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()

    model_switch_prob = 0.0
    teacher_switch_prob = 0.0
    total = 0
    for item, prob in zip(examples, probs):
        state = item["state"]
        se = _se_set_for(int(state["opp_species_gid"]))
        if not se:
            continue
        active_move = _active_move_gid(state)
        if active_move is None:
            continue
        bench = state.get("my_bench") or []
        # 生存していて相手 active に SE が通せる控えが存在するか。
        bench_has_se = any(
            slot is not None
            and float(slot.get("hp_frac", 0.0)) > 0.0
            and any(int(g) in se for g in slot.get("move_gids", []))
            for slot in bench
        )
        # active がすでに SE を通せる (= 不利対面でない) なら対象外。
        if active_move in se or not bench_has_se:
            continue
        total += 1
        model_switch_prob += sum(prob[MAX_MOVE_SLOTS:])
        target_pi = [float(x) for x in item["target_pi"]]
        teacher_switch_prob += sum(target_pi[MAX_MOVE_SLOTS:])

    if total == 0:
        return -1.0, -1.0, 0
    return model_switch_prob / total, teacher_switch_prob / total, total


# --- HP 層別 (5×5 unfold) 診断 --------------------------------------------
# stage3c の主役指標。タイプ非対称を除いても 3HKO 由来の HP 状況依存の混合戦略が
# 残るかを観察するため、交代/技選択の確率を「自分 active HP × 相手 active HP」の
# 5×5 で層別する。鏡像セルは畳まず (unfold) 別々に出し、対称均衡が学習で再現されたか
# (鏡像セルの一致) も所見にできるようにする。

# HP バケット境界 (上から): 満タン / 75%以上 / 50%以上 / 25%以上 / 25%未満。
HP_BUCKET_LABELS = ["100", ">=75", ">=50", ">=25", "<25"]


def hp_bucket(frac: float) -> int:
    """HP 割合を 5 段のバケット index (0=満タン 〜 4=瀕死寄り) へ。"""
    if frac >= 0.999:
        return 0
    if frac >= 0.75:
        return 1
    if frac >= 0.50:
        return 2
    if frac >= 0.25:
        return 3
    return 4


def hp_stratified_switch_diagnostics(
    model: nn.Module,
    encoded: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[tuple[str, int, int], tuple[float, float, int]]:
    """交代確率を (対面キー, 自分 active HP バケット, 相手 active HP バケット) で層別する。

    対面キーは `_switch_matchup_key` (active 技 × 相手種族) を流用。交代が合法で
    active が SW/BD/FS/FP のいずれかを撃てるターンのみ対象。戻り値:
      {(matchup_key, my_bucket, opp_bucket): (model_switch_prob, teacher_switch_prob, n)}
    該当サンプルのあるセルだけを含む (枯れたセルは欠測=キー無しで、呼び出し側が
    「観察不能」として扱う)。unfold (鏡像セルは別キー) で集計する。"""
    probs = _switch_probs_chunked(model, encoded, device, amp_dtype)

    model_sum: dict[tuple[str, int, int], float] = {}
    teacher_sum: dict[tuple[str, int, int], float] = {}
    counts: dict[tuple[str, int, int], int] = {}
    for item, model_p in zip(examples, probs):
        state = item["state"]
        key = _switch_matchup_key(state)
        if key is None:
            continue
        my_b = hp_bucket(float(state["my_exact_hp_frac"]))
        opp_b = hp_bucket(float(state["opp_quantized_hp_frac"]))
        cell = (key, my_b, opp_b)
        counts[cell] = counts.get(cell, 0) + 1
        model_sum[cell] = model_sum.get(cell, 0.0) + model_p
        target_pi = [float(x) for x in item["target_pi"]]
        teacher_sum[cell] = teacher_sum.get(cell, 0.0) + sum(target_pi[MAX_MOVE_SLOTS:])

    return {
        cell: (model_sum[cell] / n, teacher_sum[cell] / n, n)
        for cell, n in counts.items()
    }


def _switch_probs_chunked(
    model: nn.Module, encoded: Any, device: torch.device, amp_dtype: torch.dtype
) -> list[float]:
    """各 item の交代確率 (sum(prob[MAX_MOVE_SLOTS:])) を返す。全 item を 1 バッチで通すと
    flash-attn が巨大バッチで落ちる (CUDA invalid argument) ため、チャンク分割する。"""
    n_items = int(encoded.my_species.shape[0])
    chunk = 8192
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for start in range(0, n_items, chunk):
            sub = encoded[slice(start, min(start + chunk, n_items))]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                logits, _ = model(sub)
            p = torch.softmax(logits.float(), dim=-1)
            out.extend(p[:, MAX_MOVE_SLOTS:].sum(dim=-1).cpu().tolist())
    return out


def _bench_hp_frac(state: dict[str, Any], key: str) -> float:
    """控え 1 体目の HP 割合。空き枠 (None) や控え無しは 0.0 (瀕死バケット相当)。
    MAX_PARTY=2 前提で 1 枠のみ参照する (3v3 化時は要拡張)。"""
    lst = state.get(key) or []
    if lst and lst[0] is not None:
        return float(lst[0]["hp_frac"])
    return 0.0


def hp_4d_switch_diagnostics(
    model: nn.Module,
    encoded: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[tuple[str, int, int, int, int], tuple[float, float, int]]:
    """交代確率を (対面キー, 自分 active HP, 相手 active HP, 自分 bench HP,
    相手 bench HP) の 4 軸 (5×5×5×5) で層別する。外側 5×5 = active HP、
    内側 5×5 = bench HP のドリルダウン表示用。戻り値はサンプルのあるセルのみ:
      {(matchup, my_act_b, opp_act_b, my_bench_b, opp_bench_b): (model_p, teacher_p, n)}
    """
    probs = _switch_probs_chunked(model, encoded, device, amp_dtype)
    model_sum: dict[tuple[str, int, int, int, int], float] = {}
    teacher_sum: dict[tuple[str, int, int, int, int], float] = {}
    counts: dict[tuple[str, int, int, int, int], int] = {}
    for item, model_p in zip(examples, probs):
        state = item["state"]
        key = _switch_matchup_key(state)
        if key is None:
            continue
        cell = (
            key,
            hp_bucket(float(state["my_exact_hp_frac"])),
            hp_bucket(float(state["opp_quantized_hp_frac"])),
            hp_bucket(_bench_hp_frac(state, "my_bench")),
            hp_bucket(_bench_hp_frac(state, "opp_bench")),
        )
        counts[cell] = counts.get(cell, 0) + 1
        model_sum[cell] = model_sum.get(cell, 0.0) + model_p
        target_pi = [float(x) for x in item["target_pi"]]
        teacher_sum[cell] = teacher_sum.get(cell, 0.0) + sum(target_pi[MAX_MOVE_SLOTS:])

    return {
        cell: (model_sum[cell] / n, teacher_sum[cell] / n, n)
        for cell, n in counts.items()
    }
