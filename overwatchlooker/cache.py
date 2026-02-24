"""Disk cache for screenshot analysis results."""

import hashlib
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"


def _key(png_bytes: bytes, analyzer: str) -> str:
    h = hashlib.sha256(png_bytes).hexdigest()[:16]
    return f"{h}_{analyzer}"


def get(png_bytes: bytes, analyzer: str) -> str | None:
    path = CACHE_DIR / f"{_key(png_bytes, analyzer)}.txt"
    return path.read_text("utf-8") if path.exists() else None


def put(png_bytes: bytes, analyzer: str, result: str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{_key(png_bytes, analyzer)}.txt").write_text(result, "utf-8")
