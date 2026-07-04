#!/usr/bin/env python3
"""Drive pokemon-showdown simulate-battle CLI with two random agents.

Phase 1 sanity check: feeds team/poke-ai3/phase1/phase1_simple_team.txt
(Mew with Tackle/Strength) into pokemon-showdown's simulate-battle subprocess
and plays both sides with uniformly random move choices until a winner emerges.

Usage:
    PYTHONUNBUFFERED=1 python3 scripts/phase1_showdown_cli_battle.py
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWDOWN = REPO_ROOT / "pokemon-showdown"
TEAM_TXT = REPO_ROOT / "team" / "poke-ai3" / "phase1" / "phase1_simple_team.txt"


def pack_team(text: str) -> str:
    out = subprocess.run(
        [str(SHOWDOWN / "pokemon-showdown"), "pack-team"],
        input=text, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def main(seed: int = 12345) -> int:
    random.seed(seed)
    team_text = TEAM_TXT.read_text()
    packed = pack_team(team_text)

    proc = subprocess.Popen(
        [str(SHOWDOWN / "pokemon-showdown"), "simulate-battle"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(SHOWDOWN),
    )
    assert proc.stdin and proc.stdout

    def send(line: str) -> None:
        print(f">>> {line}", flush=True)
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    send(f'>start {{"formatid":"gen9customgame","seed":[{seed},{seed+1},{seed+2},{seed+3}]}}')
    send(f'>player p1 {{"name":"Alice","team":"{packed}"}}')
    send(f'>player p2 {{"name":"Bob","team":"{packed}"}}')

    winner: str | None = None
    pending: dict[str, dict] = {}

    def choose(side: str) -> None:
        req = pending.get(side)
        if not req:
            return
        if req.get("wait"):
            pending.pop(side, None)
            return
        if req.get("teamPreview"):
            send(f">{side} team 1")
            pending.pop(side, None)
            return
        if req.get("forceSwitch"):
            send(f">{side} default")
            pending.pop(side, None)
            return
        active = req.get("active") or []
        if not active:
            pending.pop(side, None)
            return
        moves = active[0].get("moves", [])
        legal = [i + 1 for i, m in enumerate(moves) if not m.get("disabled")]
        if not legal:
            send(f">{side} default")
        else:
            send(f">{side} move {random.choice(legal)}")
        pending.pop(side, None)

    section: str | None = None
    side_label: str | None = None
    buf: list[str] = []

    def flush_block() -> None:
        nonlocal section, side_label, buf, winner
        if section is None:
            return
        for line in buf:
            print(f"[{section}{':'+side_label if side_label else ''}] {line}", flush=True)
            if line.startswith("|request|"):
                payload = line[len("|request|"):]
                if payload and side_label:
                    try:
                        pending[side_label] = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
            elif line.startswith("|win|"):
                winner = line[len("|win|"):]
            elif line.startswith("|tie|"):
                winner = "(tie)"
        section = None
        side_label = None
        buf = []

    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if line == "":
            flush_block()
            for s in ("p1", "p2"):
                choose(s)
            if winner is not None:
                break
            continue
        if section is None:
            section = line
            continue
        if section == "sideupdate" and side_label is None:
            side_label = line
            continue
        buf.append(line)

    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    err = proc.stderr.read() if proc.stderr else ""
    if err:
        print("STDERR:", err, file=sys.stderr)
    print(f"\n=== WINNER: {winner} ===", flush=True)
    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
