"""Disk cache for screenshot analysis results."""

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"


def _key(png_bytes: bytes, analyzer: str) -> str:
    h = hashlib.sha256(png_bytes).hexdigest()[:16]
    return f"{h}_{analyzer}"


def get(png_bytes: bytes, analyzer: str) -> dict | str | None:
    """Return cached result: dict (JSON) or str (legacy text), or None."""
    # Try JSON first (new format)
    json_path = CACHE_DIR / f"{_key(png_bytes, analyzer)}.json"
    if json_path.exists():
        return json.loads(json_path.read_text("utf-8"))
    # Fall back to legacy text cache
    txt_path = CACHE_DIR / f"{_key(png_bytes, analyzer)}.txt"
    if txt_path.exists():
        return txt_path.read_text("utf-8")
    return None


def put(png_bytes: bytes, analyzer: str, result: dict | str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    if isinstance(result, dict):
        (CACHE_DIR / f"{_key(png_bytes, analyzer)}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
        )
    else:
        (CACHE_DIR / f"{_key(png_bytes, analyzer)}.txt").write_text(result, "utf-8")
