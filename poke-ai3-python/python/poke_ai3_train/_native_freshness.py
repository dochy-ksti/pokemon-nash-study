from __future__ import annotations

import os
import sys
from pathlib import Path

# uv run の再ビルド判定は poke-ai3-python/ 配下の変更しか見ないため、ワークスペース
# 依存先 (poke-env-rust / poke-sho-rust) を編集しても _native.so が再生成されず古いまま
# 残ることがある。起動前に Rust ソースと _native.so の mtime を比べ、古ければ停止する。
# 回避したいときは POKE_AI3_SKIP_FRESH_CHECK=1 を設定する。

_REBUILD_CMD = "cd poke-ai3-python && uv run maturin develop --release"


def _newest_rust_mtime(repo_root: Path) -> tuple[float, Path | None]:
    """ワークスペース内 Rust ソース (*.rs / Cargo.toml) の最新 mtime とそのパス。"""
    crates = ["poke-ai3-python", "poke-env-rust", "poke-sho-rust"]
    newest = 0.0
    newest_path: Path | None = None
    candidates: list[Path] = [repo_root / "Cargo.toml"]
    for crate in crates:
        base = repo_root / crate
        candidates.append(base / "Cargo.toml")
        candidates.extend((base / "src").rglob("*.rs"))
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > newest:
            newest = mtime
            newest_path = path
    return newest, newest_path


def _native_so(repo_root: Path) -> Path | None:
    pkg = repo_root / "poke-ai3-python" / "python" / "poke_ai3"
    sos = sorted(pkg.glob("_native*.so"))
    return sos[0] if sos else None


def check_native_fresh() -> None:
    """_native.so が最新 Rust ソースより古ければ警告して終了する。"""
    if os.environ.get("POKE_AI3_SKIP_FRESH_CHECK"):
        return
    repo_root = Path(__file__).resolve().parents[3]
    so = _native_so(repo_root)
    if so is None:
        return  # 未ビルド。import 時の通常エラーに委ねる。
    so_mtime = so.stat().st_mtime
    rust_mtime, rust_path = _newest_rust_mtime(repo_root)
    if rust_mtime <= so_mtime:
        return
    rel = rust_path.relative_to(repo_root) if rust_path else "(unknown)"
    sys.stderr.write(
        "\n[poke-ai3] エラー: Rust の _native.so が最新ソースより古いです。\n"
        f"  _native.so : {so.name} ({_fmt(so_mtime)})\n"
        f"  最新ソース : {rel} ({_fmt(rust_mtime)})\n"
        "  uv run の再ビルド判定はワークスペース依存先 (poke-env-rust 等) の変更を\n"
        "  検知しないため、明示的に再ビルドしてください:\n"
        f"    {_REBUILD_CMD}\n"
        "  どうしても無視して起動する場合は POKE_AI3_SKIP_FRESH_CHECK=1 を設定。\n\n"
    )
    raise SystemExit(1)


def _fmt(mtime: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
