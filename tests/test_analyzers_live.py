"""Live LLM analyzer tests — opt-in only, run with: pytest -m live

These tests make real API calls. Codex is subscription-based so safe to call.
"""

import pytest

from tests.conftest import FIXTURES_DIR


pytestmark = pytest.mark.live


@pytest.fixture
def tab_png():
    path = FIXTURES_DIR / "2026-03-07_21-33-27.png"
    if not path.exists():
        pytest.skip("Test fixture screenshots not found")
    return path.read_bytes()


@pytest.fixture
def non_tab_png():
    path = FIXTURES_DIR / "2026-03-11_22-15-07.png"
    if not path.exists():
        pytest.skip("Test fixture screenshots not found")
    return path.read_bytes()


@pytest.fixture
def hero_crops():
    """Two hero panel crops for testing multi-hero analysis."""
    from overwatchlooker.screenshot import crop_hero_panel

    crops = {}
    for filename, hero in [("2026-03-07_21-00-39.png", "Reinhardt"),
                           ("2026-03-11_23-37-05.png", "Moira")]:
        path = FIXTURES_DIR / filename
        if not path.exists():
            pytest.skip("Test fixture screenshots not found")
        crops[hero] = crop_hero_panel(path.read_bytes())
    return crops


class TestCodexLive:
    def test_basic_tab(self, tab_png):
        from overwatchlooker.analyzers.codex import analyze_screenshot

        result = analyze_screenshot(tab_png, audio_result="VICTORY")

        assert isinstance(result, dict)
        assert result["not_ow2_tab"] is False
        assert result["map_name"]
        assert result["mode"] in ("PUSH", "CONTROL", "ESCORT", "HYBRID", "CLASH", "FLASHPOINT")
        assert result["queue_type"] in ("COMPETITIVE", "QUICKPLAY")
        assert len(result["players"]) == 10
        assert any(p["is_self"] for p in result["players"])

    def test_with_hero_crops(self, tab_png, hero_crops):
        from overwatchlooker.analyzers.codex import analyze_screenshot

        result = analyze_screenshot(tab_png, audio_result="VICTORY", hero_crops=hero_crops)

        assert isinstance(result, dict)
        assert "extra_hero_stats" in result
        assert len(result["extra_hero_stats"]) >= 1
        for entry in result["extra_hero_stats"]:
            assert "hero_name" in entry
            assert "stats" in entry

    def test_non_tab_rejection(self, non_tab_png):
        from overwatchlooker.analyzers.codex import analyze_screenshot

        result = analyze_screenshot(non_tab_png)

        assert isinstance(result, dict)
        # The match-end screen may or may not be detected as tab by LLM,
        # but it should at least return a valid dict
        assert "not_ow2_tab" in result
