"""人間 vs AI 対戦 GUI のバックエンド (Phase1: ネット単体)。

ブラウザ (単一 HTML+JS) ↔ HTTP/JSON ↔ 本サーバ (torch Agent 保持) ↔ pyo3 ↔ Rust
(`HumanGame`)。学習済みチェックポイントをロードし、毎ターン AI の policy 分布と
value (盤面勝率) を返して可視化する。AI の着手は policy からのサンプリング (Q6=B)。
ダメージは決定論 (Rust 側 MaxRoll)、行動順コインのみ毎ターン公平なランダム。

起動: `uv run play-gui --port 8000 --checkpoint data/poke-ai3/stage3b_long_xteam.pt`
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from .agent import Agent
from poke_ai3 import ACTION_DIM, MAX_MOVE_SLOTS

from .encoding import encode_observations

ASSETS_DIR = Path(__file__).resolve().parent / "play_gui_assets"


def build_native_extension() -> None:
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-m", "maturin", "develop", "--quiet"],
        cwd=project_root,
        check=True,
    )


def load_human_game_cls() -> Any:
    build_native_extension()
    return importlib.import_module("poke_ai3").HumanGame


def ai_policy_value(agent: Agent, obs: dict[str, Any]) -> tuple[list[float], float]:
    """AI 視点観測から softmax policy (長さ ACTION_DIM) と value (勝率 0..1) を返す。"""
    encoded = encode_observations([{"state": obs}], agent.device)
    agent.model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=agent.device.type,
            dtype=agent.agent_config.amp_dtype,
            enabled=agent.device.type == "cuda",
        ):
            logits, values = agent.model(encoded)
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()[0]
        raw = values.float().cpu().tolist()[0]
    value = float(raw[0]) if isinstance(raw, list) else float(raw)
    return [float(p) for p in probs], value


def sample_action(probs: list[float], legal_mask: list[bool]) -> int:
    """policy を合法手に制限してサンプリングする。"""
    weights = [p if (i < len(legal_mask) and legal_mask[i]) else 0.0 for i, p in enumerate(probs)]
    total = sum(weights)
    legal_idx = [i for i, m in enumerate(legal_mask) if m]
    if total <= 0.0:
        return random.choice(legal_idx) if legal_idx else 0
    r = random.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r < acc:
            return i
    return legal_idx[-1] if legal_idx else 0


class Session:
    """単一の進行中対戦。状態の真実は Rust 側 `HumanGame` が保持する。"""

    def __init__(self, agent: Agent, game_cls: Any, stage: str) -> None:
        self.agent = agent
        self.game_cls = game_cls
        self.stage = stage

        self.game: Any | None = None

    def new(self, human_team: str, human_lead: int, ai_lead: int) -> dict[str, Any]:
        self.game = self.game_cls(human_team, human_lead, ai_lead, self.stage)
        return self.state()

    def state(self) -> dict[str, Any]:
        assert self.game is not None
        human_obs = json.loads(self.game.observation(1))
        ai_obs = json.loads(self.game.observation(2))
        probs, value = ai_policy_value(self.agent, ai_obs)
        return {
            "human": human_obs,      # 人間 (P1) 視点 (相手は種族 + HP% のみ)
            "ai_view": ai_obs,       # 神視点トグル用 (AI 自身の全情報 = 相手の手の内)
            "ai_policy": probs,      # AI の policy 分布 (長さ ACTION_DIM)
            "ai_value": value,       # AI の盤面勝率予測 (0..1, AI 自身視点)
            "turn": self.game.turn(),
            "done": self.game.done(),
            "winner": self.game.winner(),
            "draw": self.game.is_draw(),
            "num_moves": MAX_MOVE_SLOTS,  # 行動 index < num_moves は技スロット、以降は交代。
            "stage": self.stage,
        }

    def action(self, human_action: int) -> dict[str, Any]:
        assert self.game is not None
        ai_obs = json.loads(self.game.observation(2))
        probs, _ = ai_policy_value(self.agent, ai_obs)
        ai_action = sample_action(probs, ai_obs["legal_action_mask"])
        human_first = random.random() < 0.5
        result = json.loads(self.game.step(human_action, ai_action, human_first))
        result["ai_action"] = ai_action
        result["human_action"] = human_action
        # 終局でなければ次ターンの観測・AI 評価も返す。
        if not self.game.done():
            result["next"] = self.state()
        else:
            result["next"] = self.state()
        return result


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # noqa: D401 - 静かに
            pass

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send_file(ASSETS_DIR / "index.html", "text/html; charset=utf-8")
            elif self.path == "/state":
                if session.game is None:
                    self._send_json({"error": "no game"}, status=400)
                else:
                    self._send_json(session.state())
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            try:
                body = self._read_body()
                if self.path == "/new":
                    payload = session.new(
                        body.get("human_team", "team1"),
                        int(body.get("human_lead", 0)),
                        int(body.get("ai_lead", 0)),
                    )
                    self._send_json(payload)
                elif self.path == "/action":
                    if session.game is None or session.game.done():
                        self._send_json({"error": "no active game"}, status=400)
                        return
                    self._send_json(session.action(int(body["action"])))
                else:
                    self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # noqa: BLE001 - GUI なので握りつぶしてメッセージ返却
                self._send_json({"error": str(exc)}, status=500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="人間 vs AI 対戦 GUI サーバ")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/poke-ai3/stage3b_long_xteam.pt"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage", default="3b", choices=["3b", "3c"])
    args = parser.parse_args()

    game_cls = load_human_game_cls()
    print(f"loading checkpoint: {args.checkpoint}  (stage={args.stage})")
    agent = Agent(device=args.device, checkpoint_path=args.checkpoint)
    session = Session(agent, game_cls, args.stage)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"play-gui ready: http://{args.host}:{args.port}  (ACTION_DIM={ACTION_DIM}, MAX_MOVE_SLOTS={MAX_MOVE_SLOTS})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        server.shutdown()


if __name__ == "__main__":
    main()
