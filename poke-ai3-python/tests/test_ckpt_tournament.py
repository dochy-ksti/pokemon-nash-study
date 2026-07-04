"""AllHistoryPeakDetector の選抜ロジック回帰テスト (実対戦なし、winrate を注入)。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ckpt_tournament", Path(__file__).resolve().parents[1] / "scripts" / "ckpt_tournament.py"
)
ckpt_tournament = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ckpt_tournament)
AllHistoryPeakDetector = ckpt_tournament.AllHistoryPeakDetector


def _detector(winrate):
    discarded: list[Path] = []
    det = AllHistoryPeakDetector(
        winrate=winrate,
        pick_winner=lambda pool: pool[0],  # テストでは先頭を勝者扱い
        discard=discarded.append,
    )
    return det, discarded


def test_monotonic_phase_never_peaks() -> None:
    # 新は常に全履歴に勝つ (winrate>0.5) → ピーク無し、history が伸び続ける。
    det, discarded = _detector(lambda new, old: 0.7)
    for i in range(5):
        assert det.feed(Path(f"ep{i}.pt")) is None
    assert len(det.history) == 5
    assert discarded == []


def test_cycle_emits_peak_and_resets() -> None:
    items = [Path(f"ep{i}.pt") for i in range(4)]
    # ep0..ep2 は上り坂。ep3 は ep0 にだけ負ける (循環)。
    def winrate(new: Path, old: Path) -> float:
        if new == items[3] and old == items[0]:
            return 0.3  # 最新が最古に敗北
        return 0.6
    det, discarded = _detector(winrate)
    assert det.feed(items[0]) is None
    assert det.feed(items[1]) is None
    assert det.feed(items[2]) is None
    peak = det.feed(items[3])
    assert peak == items[0]            # 最新を倒した古 snapshot が昇格
    assert det.history == []           # リセットで空
    # peak 以外の history (ep1, ep2) と 敗北した最新 (ep3) は削除。
    assert set(discarded) == {items[1], items[2], items[3]}


def test_multiple_beaters_go_to_round_robin() -> None:
    items = [Path(f"ep{i}.pt") for i in range(3)]
    # 最新(ep2)は ep0, ep1 の両方に負ける → pick_winner(先頭=ep0) が昇格。
    det, discarded = _detector(lambda new, old: 0.4 if new == items[2] else 0.6)
    det.feed(items[0])
    det.feed(items[1])
    peak = det.feed(items[2])
    assert peak == items[0]
    assert set(discarded) == {items[1], items[2]}


# ---------------------------------------------------------------- block 学習化

import json
import re
import types


def _args(tmp_path, tag="A", **over):
    ns = types.SimpleNamespace(
        tag=tag, start=None, resume=False,
        epochs_per_step=5, train_block_epochs=50,
        warmup_steps=1, peaks_per_rr=1, finalists_target=1,
        max_added_epochs=1000, depth_skew=1.0,
        search_turn_min=4, search_turn_max=8,
        # winrate_vs / round_robin_winner / train_block_to は stub するので不問。
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _install_stubs(monkeypatch, tmp_path, winrate, *, train=True):
    """TDIR を tmp に向け、train_block_to/winrate_vs/round_robin_winner を stub する。"""
    monkeypatch.setattr(ckpt_tournament, "TDIR", tmp_path)
    fed: list[int] = []

    def wr_stub(new, old, args):
        e = int(re.search(r"_ep(\d+)\.pt$", new.name).group(1))
        return winrate(new, old)

    monkeypatch.setattr(ckpt_tournament, "winrate_vs", wr_stub)
    monkeypatch.setattr(ckpt_tournament, "round_robin_winner",
                        lambda pool, args: pool[0])

    def train_stub(work, block_target, args):
        # train-loop の --snapshot-every 相当: 現フロンティアから epochs_per_step 刻みで
        # block_target までダミー snapshot を作る。
        existing = [int(m.group(1)) for p in work.parent.glob(f"{work.stem}_ep*.pt")
                    if (m := re.search(r"_ep(\d+)\.pt$", p.name))]
        frontier = max(existing) if existing else 0
        e = frontier + args.epochs_per_step
        while e <= block_target:
            (work.parent / f"{work.stem}_ep{e}.pt").write_text("dummy")
            e += args.epochs_per_step

    if train:
        monkeypatch.setattr(ckpt_tournament, "train_block_to", train_stub)
    return fed


def test_list_snapshots_order_and_range(tmp_path) -> None:
    work = tmp_path / "A.pt"
    for e in (5, 10, 20, 15):
        (tmp_path / f"A_ep{e}.pt").write_text("x")
    (tmp_path / "A.pt").write_text("work")  # 作業用は除外される
    got = ckpt_tournament.list_snapshots(work, 5, 20)
    assert [e for e, _ in got] == [10, 15, 20]  # 昇順 & lo<e<=hi


def test_block_warmup_deleted_and_no_peak(tmp_path, monkeypatch) -> None:
    # 全 snapshot が履歴に勝つ (winrate 0.6) → ピーク無し。
    # finalists_target に到達しないよう max_added_epochs を小さくして打ち切る。
    _install_stubs(monkeypatch, tmp_path, lambda new, old: 0.6)
    args = _args(tmp_path, finalists_target=99, max_added_epochs=50, train_block_epochs=50)
    ckpt_tournament.funnel(args)
    # warmup (ep5) は削除済み、ep10..50 は history としてファイル保持。
    assert not (tmp_path / "A_ep5.pt").exists()
    for e in range(10, 51, 5):
        assert (tmp_path / f"A_ep{e}.pt").exists()


def test_finalists_target_stops_and_keeps_surplus(tmp_path, monkeypatch) -> None:
    # ep10 を history へ、ep15 が ep10 に敗北 → ep10 が finalist。target=1 で停止。
    def winrate(new, old):
        return 0.4 if new.name.endswith("_ep15.pt") else 0.6
    _install_stubs(monkeypatch, tmp_path, winrate)
    args = _args(tmp_path, finalists_target=1, peaks_per_rr=1, train_block_epochs=50)
    ckpt_tournament.funnel(args)
    out = json.loads((tmp_path / "A_finalists.json").read_text())
    assert out["finalists"] == [str(tmp_path / "A_ep10.pt")]
    assert (tmp_path / "A_ep10.pt").exists()          # finalist は保護
    assert not (tmp_path / "A_ep15.pt").exists()      # 敗者は det.feed (#2) が削除
    # #5 撤廃: finalists_target 到達後の余剰 snapshot は削除せず run 終了まで残す。
    for e in range(20, 51, 5):
        assert (tmp_path / f"A_ep{e}.pt").exists()


def test_resume_consumes_unprocessed(tmp_path, monkeypatch) -> None:
    # 中断状態: ep10 まで処理済み (history), ep15..50 が未処理で残存。
    feed_order: list[int] = []

    def winrate(new, old):
        feed_order.append(int(re.search(r"_ep(\d+)\.pt$", new.name).group(1)))
        return 0.6  # 全勝 → ピーク無し、ただ順に消費されるだけ

    _install_stubs(monkeypatch, tmp_path, winrate)
    work = tmp_path / "A.pt"
    work.write_text("work")
    for e in range(10, 51, 5):
        (tmp_path / f"A_ep{e}.pt").write_text("dummy")
    (tmp_path / "A_state.json").write_text(json.dumps({
        "base": 0, "epoch": 10, "warmup_until": 5, "block_target": 50,
        "history": [str(tmp_path / "A_ep10.pt")], "peaks": [], "finalists": [],
    }))
    args = _args(tmp_path, resume=True, finalists_target=99,
                 max_added_epochs=50, train_block_epochs=50)
    ckpt_tournament.funnel(args)
    # 未処理の ep15..50 のみが昇順で feed される (ep10 は処理済みなので除外)。
    # winrate は履歴比較ごとに呼ばれるため、消費順は出現順の重複排除で見る。
    seen: list[int] = []
    for e in feed_order:
        if e not in seen:
            seen.append(e)
    assert seen == [15, 20, 25, 30, 35, 40, 45, 50]
