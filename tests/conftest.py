"""Shared fixtures for OverwatchLooker test suite."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Screenshot fixtures: (filename, has_hero_panel, expected_hero_name)
SCREENSHOT_FIXTURES = [
    ("2026-03-07_21-33-27.png", True, "Juno"),
    ("2026-03-07_21-37-05.png", True, "Juno"),
    ("2026-03-07_21-41-45.png", True, "Juno"),
    ("2026-03-07_21-00-39.png", True, "Reinhardt"),
    ("2026-03-11_22-15-07.png", False, None),
    ("2026-03-11_23-37-05.png", True, "Moira"),
]


def _have_fixtures() -> bool:
    return all((FIXTURES_DIR / f).exists() for f, _, _ in SCREENSHOT_FIXTURES)


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless explicitly requested via -m live."""
    run_live = False
    markexpr = config.getoption("-m", default="")
    if "live" in markexpr:
        run_live = True

    for item in items:
        if "live" in item.keywords and not run_live:
            item.add_marker(pytest.mark.skip(reason="live tests require -m live"))


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory, skip if missing."""
    if not _have_fixtures():
        pytest.skip("Test fixture screenshots not found")
    return FIXTURES_DIR


@pytest.fixture
def sample_tab_png(fixtures_dir):
    """A real OW2 Tab screenshot as bytes."""
    return (fixtures_dir / "2026-03-07_21-33-27.png").read_bytes()


@pytest.fixture
def sample_non_tab_png():
    """A synthetic non-Tab image (random noise, no scoreboard structure)."""
    import cv2
    import numpy as np
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture
def sample_hero_panel_png(fixtures_dir):
    """A Tab screenshot that has a visible hero panel."""
    return (fixtures_dir / "2026-03-07_21-00-39.png").read_bytes()


@pytest.fixture
def sample_no_panel_png(fixtures_dir):
    """A screenshot without a hero stats panel."""
    return (fixtures_dir / "2026-03-11_22-15-07.png").read_bytes()


@pytest.fixture
def sample_match_dict():
    """A realistic match dict matching MATCH_SCHEMA."""
    return {
        "not_ow2_tab": False,
        "map_name": "Lijiang Tower",
        "duration": "8:06",
        "mode": "CONTROL",
        "queue_type": "COMPETITIVE",
        "result": "VICTORY",
        "players": [
            {
                "team": "ALLY", "role": "TANK",
                "player_name": "PLAYER1", "title": "Stalwart Hero",
                "eliminations": 15, "assists": 8, "deaths": 3,
                "damage": 12500, "healing": 0, "mitigation": 8900,
                "is_self": True,
                "hero": {
                    "hero_name": "Reinhardt",
                    "stats": [
                        {"label": "Charge Kills", "value": "3", "is_featured": True},
                        {"label": "Fire Strike Kills", "value": "2", "is_featured": False},
                    ],
                },
            },
            {
                "team": "ALLY", "role": "DPS",
                "player_name": "PLAYER2", "title": None,
                "eliminations": 20, "assists": 5, "deaths": 4,
                "damage": 15000, "healing": 0, "mitigation": 0,
                "is_self": False, "hero": None,
            },
            {
                "team": "ALLY", "role": "DPS",
                "player_name": "PLAYER3", "title": None,
                "eliminations": 18, "assists": 6, "deaths": 5,
                "damage": 13000, "healing": 0, "mitigation": 0,
                "is_self": False, "hero": None,
            },
            {
                "team": "ALLY", "role": "SUPPORT",
                "player_name": "PLAYER4", "title": None,
                "eliminations": 5, "assists": 20, "deaths": 2,
                "damage": 3000, "healing": 12000, "mitigation": 0,
                "is_self": False, "hero": None,
            },
            {
                "team": "ALLY", "role": "SUPPORT",
                "player_name": "PLAYER5", "title": None,
                "eliminations": 7, "assists": 18, "deaths": 3,
                "damage": 4000, "healing": 10000, "mitigation": 0,
                "is_self": False, "hero": None,
            },
            {
                "team": "ENEMY", "role": "TANK",
                "player_name": "ENEMY1", "title": None,
                "eliminations": 10, "assists": 5, "deaths": 6,
                "damage": 9000, "healing": 0, "mitigation": 7000,
                "is_self": False, "hero": None,
            },
            {
                "team": "ENEMY", "role": "DPS",
                "player_name": "ENEMY2", "title": None,
                "eliminations": 12, "assists": 3, "deaths": 7,
                "damage": 11000, "healing": 0, "mitigation": 0,
                "is_self": False, "hero": None,
            },
            {
                "team": "ENEMY", "role": "DPS",
                "player_name": "ENEMY3", "title": None,
                "eliminations": 8, "assists": 4, "deaths": 8,
                "damage": 8000, "healing": 0, "mitigation": 0,
                "is_self": False, "hero": None,
            },
            {
                "team": "ENEMY", "role": "SUPPORT",
                "player_name": "ENEMY4", "title": None,
                "eliminations": 3, "assists": 15, "deaths": 5,
                "damage": 2000, "healing": 9000, "mitigation": 0,
                "is_self": False, "hero": None,
            },
            {
                "team": "ENEMY", "role": "SUPPORT",
                "player_name": "ENEMY5", "title": None,
                "eliminations": 4, "assists": 12, "deaths": 6,
                "damage": 2500, "healing": 8000, "mitigation": 0,
                "is_self": False, "hero": None,
            },
        ],
    }
