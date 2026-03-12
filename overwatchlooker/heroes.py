"""Shared hero name list, edit distance, and fuzzy matching."""

import logging
from pathlib import Path

_logger = logging.getLogger("overwatchlooker")

_HEROES_FILE = Path(__file__).parent / "heroes.txt"

ALL_HEROES: list[str] = []


def load_heroes() -> None:
    """Load hero names from heroes.txt. Called once at app start."""
    global ALL_HEROES
    ALL_HEROES = [
        line.strip()
        for line in _HEROES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _logger.info(f"Loaded {len(ALL_HEROES)} hero names from {_HEROES_FILE.name}")


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        return edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def match_hero_name(raw_text: str) -> str:
    """Fuzzy-match raw OCR text to the closest known hero name.

    Returns the hero name in proper case, or empty string if no good match.
    """
    raw = raw_text.lower().strip()
    if not raw:
        return ""
    best_hero, best_dist = "", 999
    for hero in ALL_HEROES:
        d = edit_distance(raw, hero.lower().replace(" ", ""))
        if d < best_dist:
            best_dist = d
            best_hero = hero
    if best_dist <= max(2, len(best_hero) * 0.4):
        return best_hero
    return ""


# Load on import so heroes are available immediately
load_heroes()
