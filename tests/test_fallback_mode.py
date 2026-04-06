"""Tests for fallback mode: LLM analysis integration, MatchState population."""

from unittest.mock import MagicMock, patch

import pytest

from overwatchlooker.match_state import (
    HeroSource,
    MatchResult,
    MatchState,
    PlayerRole,
    PlayerState,
    ResultSource,
    TeamSide,
    build_mcp_payload,
)
from overwatchlooker.overwolf import FallbackModeEvent, GameType


# Sample LLM response matching the schema
SAMPLE_LLM_RESPONSE = {
    "not_ow2_tab": False,
    "map_name": "King's Row",
    "duration": "8:24",
    "mode": "HYBRID",
    "queue_type": "COMPETITIVE",
    "players": [
        {
            "team": "ALLY", "role": "TANK", "player_name": "PLAYER1",
            "title": None, "hero_name": None,
            "eliminations": 15, "assists": 8, "deaths": 4,
            "damage": 12500, "healing": 0, "mitigation": 8900,
            "is_self": False,
        },
        {
            "team": "ALLY", "role": "DPS", "player_name": "TESTPLAYER",
            "title": "Stalwart Hero", "hero_name": "Cassidy",
            "eliminations": 22, "assists": 5, "deaths": 6,
            "damage": 18000, "healing": 0, "mitigation": 0,
            "is_self": True,
        },
        {
            "team": "ALLY", "role": "DPS", "player_name": "PLAYER3",
            "title": None, "hero_name": None,
            "eliminations": 18, "assists": 3, "deaths": 5,
            "damage": 15000, "healing": 0, "mitigation": 0,
            "is_self": False,
        },
        {
            "team": "ALLY", "role": "SUPPORT", "player_name": "PLAYER4",
            "title": None, "hero_name": None,
            "eliminations": 5, "assists": 20, "deaths": 3,
            "damage": 3000, "healing": 12000, "mitigation": 0,
            "is_self": False,
        },
        {
            "team": "ALLY", "role": "SUPPORT", "player_name": "PLAYER5",
            "title": None, "hero_name": None,
            "eliminations": 4, "assists": 18, "deaths": 4,
            "damage": 2500, "healing": 10000, "mitigation": 0,
            "is_self": False,
        },
        {
            "team": "ENEMY", "role": "TANK", "player_name": "ENEMY1",
            "title": None, "hero_name": None,
            "eliminations": 10, "assists": 6, "deaths": 8,
            "damage": 9000, "healing": 0, "mitigation": 7500,
            "is_self": False,
        },
        {
            "team": "ENEMY", "role": "DPS", "player_name": "ENEMY2",
            "title": None, "hero_name": None,
            "eliminations": 12, "assists": 4, "deaths": 7,
            "damage": 14000, "healing": 0, "mitigation": 0,
            "is_self": False,
        },
        {
            "team": "ENEMY", "role": "DPS", "player_name": "ENEMY3",
            "title": None, "hero_name": None,
            "eliminations": 14, "assists": 5, "deaths": 6,
            "damage": 16000, "healing": 0, "mitigation": 0,
            "is_self": False,
        },
        {
            "team": "ENEMY", "role": "SUPPORT", "player_name": "ENEMY4",
            "title": None, "hero_name": None,
            "eliminations": 3, "assists": 15, "deaths": 5,
            "damage": 2000, "healing": 11000, "mitigation": 0,
            "is_self": False,
        },
        {
            "team": "ENEMY", "role": "SUPPORT", "player_name": "ENEMY5",
            "title": None, "hero_name": None,
            "eliminations": 2, "assists": 12, "deaths": 6,
            "damage": 1800, "healing": 9000, "mitigation": 0,
            "is_self": False,
        },
    ],
}


@pytest.fixture
def fallback_app():
    """Create an App instance with fallback mode enabled."""
    from overwatchlooker.tray import App
    a = App(fallback=True)
    return a


class TestFallbackModeFlag:
    def test_fallback_mode_on_init(self, fallback_app):
        assert fallback_app._fallback_mode is True

    def test_fallback_mode_off_by_default(self):
        from overwatchlooker.tray import App
        a = App()
        assert a._fallback_mode is False

    def test_set_fallback_mode_toggles(self):
        """WS-enabled fallback can be toggled off, CLI fallback cannot."""
        from overwatchlooker.tray import App
        a = App()  # no CLI fallback
        a._set_fallback_mode(True)
        assert a._fallback_mode is True
        a._set_fallback_mode(False)
        assert a._fallback_mode is False

    def test_cli_fallback_cannot_be_disabled(self, fallback_app):
        """--fallback flag makes fallback permanent."""
        fallback_app._set_fallback_mode(False)
        assert fallback_app._fallback_mode is True

    def test_set_fallback_mode_idempotent(self, fallback_app):
        """Setting same value does nothing."""
        fallback_app._set_fallback_mode(True)
        assert fallback_app._fallback_mode is True


class TestFallbackModeEvent:
    def test_fallback_mode_event_dataclass(self):
        event = FallbackModeEvent(enabled=True)
        assert event.enabled is True

    def test_overwolf_event_dispatches_fallback(self, fallback_app):
        """FallbackModeEvent should toggle fallback mode."""
        fallback_app._fallback_mode = False
        fallback_app._on_overwolf_event(FallbackModeEvent(enabled=True))
        assert fallback_app._fallback_mode is True


class TestPopulateFromLlm:
    def test_populates_map_mode_duration(self, fallback_app):
        snapshot = MatchState()
        snapshot.latest_tab = MagicMock(png_bytes=b"fake")
        snapshot.result = MatchResult.VICTORY
        snapshot.ended_at = 600000  # need ended_at for duration computation

        with patch("overwatchlooker.llm_analyzer.analyze_tab_screenshot",
                   return_value=SAMPLE_LLM_RESPONSE):
            fallback_app._run_llm_analysis(snapshot, {})

        assert snapshot.map_name == "King's Row"
        assert snapshot.mode == "HYBRID"
        # Duration = ended_at - started_at; started_at = ended_at - (8*60+24)*1000
        assert snapshot.duration == 8 * 60000 + 24 * 1000
        assert snapshot.game_type == GameType.RANKED

    def test_populates_players(self, fallback_app):
        snapshot = MatchState()
        snapshot.latest_tab = MagicMock(png_bytes=b"fake")

        with patch("overwatchlooker.llm_analyzer.analyze_tab_screenshot",
                   return_value=SAMPLE_LLM_RESPONSE):
            fallback_app._run_llm_analysis(snapshot, {})

        assert len(snapshot.players) == 10
        # Check self player
        self_player = snapshot.players.get("TESTPLAYER")
        assert self_player is not None
        assert self_player.is_local is True
        assert self_player.role == PlayerRole.DAMAGE
        assert self_player.team_side == TeamSide.ALLY
        assert self_player.stats is not None
        assert self_player.stats.kills == 22
        assert self_player.stats.damage == 18000.0
        # Hero from LLM panel
        assert self_player.current_hero == "Cassidy"

    def test_populates_enemy_team(self, fallback_app):
        snapshot = MatchState()
        snapshot.latest_tab = MagicMock(png_bytes=b"fake")

        with patch("overwatchlooker.llm_analyzer.analyze_tab_screenshot",
                   return_value=SAMPLE_LLM_RESPONSE):
            fallback_app._run_llm_analysis(snapshot, {})

        enemies = [p for p in snapshot.players.values()
                   if p.team_side == TeamSide.ENEMY]
        assert len(enemies) == 5

    def test_not_ow2_tab_skips(self, fallback_app):
        snapshot = MatchState()
        snapshot.latest_tab = MagicMock(png_bytes=b"fake")

        response = {**SAMPLE_LLM_RESPONSE, "not_ow2_tab": True}
        with patch("overwatchlooker.llm_analyzer.analyze_tab_screenshot",
                   return_value=response):
            fallback_app._run_llm_analysis(snapshot, {})

        assert len(snapshot.players) == 0

    def test_does_not_overwrite_existing_players(self, fallback_app):
        """If Overwolf already populated players, LLM doesn't overwrite."""
        snapshot = MatchState()
        snapshot.latest_tab = MagicMock(png_bytes=b"fake")
        snapshot.players["EXISTING#1234"] = PlayerState(
            player_name="EXISTING#1234", battletag="EXISTING#1234")

        with patch("overwatchlooker.llm_analyzer.analyze_tab_screenshot",
                   return_value=SAMPLE_LLM_RESPONSE):
            fallback_app._run_llm_analysis(snapshot, {})

        # Should keep existing player, not add LLM players
        assert "EXISTING#1234" in snapshot.players
        assert len(snapshot.players) == 1

    def test_quickplay_detection(self, fallback_app):
        snapshot = MatchState()
        snapshot.latest_tab = MagicMock(png_bytes=b"fake")

        response = {**SAMPLE_LLM_RESPONSE, "queue_type": "QUICKPLAY"}
        with patch("overwatchlooker.llm_analyzer.analyze_tab_screenshot",
                   return_value=response):
            fallback_app._run_llm_analysis(snapshot, {})

        assert snapshot.game_type == GameType.UNRANKED

    def test_merges_subtitle_hero_history(self, fallback_app):
        """Subtitle hero_history should override LLM hero data."""
        snapshot = MatchState()
        snapshot.latest_tab = MagicMock(png_bytes=b"fake")

        hero_history = {
            "TESTPLAYER": [(10.0, "Tracer"), (120.0, "Cassidy")],
        }

        with patch("overwatchlooker.llm_analyzer.analyze_tab_screenshot",
                   return_value=SAMPLE_LLM_RESPONSE):
            fallback_app._run_llm_analysis(snapshot, hero_history)

        self_player = snapshot.players["TESTPLAYER"]
        assert len(self_player.hero_swaps) == 2
        assert self_player.hero_swaps[0].hero == "Tracer"
        assert self_player.hero_swaps[1].hero == "Cassidy"
        assert self_player.hero_swaps[0].source == HeroSource.SUBTITLE_OCR


class TestFallbackMcpPayload:
    def test_payload_from_llm_data(self, fallback_app):
        """Verify build_mcp_payload works with LLM-populated MatchState."""
        snapshot = MatchState()
        snapshot.latest_tab = MagicMock(png_bytes=b"fake")
        snapshot.result = MatchResult.VICTORY
        snapshot.result_source = ResultSource.SUBTITLE
        snapshot.ended_at = 500000

        with patch("overwatchlooker.llm_analyzer.analyze_tab_screenshot",
                   return_value=SAMPLE_LLM_RESPONSE):
            fallback_app._run_llm_analysis(snapshot, {})

        payload = build_mcp_payload(snapshot)
        assert payload["map_name"] == "King's Row"
        assert payload["mode"] == "HYBRID"
        assert payload["queue_type"] == "COMPETITIVE"
        assert payload["result"] == "VICTORY"
        assert len(payload["players"]) == 10

        # Self player should have stats
        self_p = next(p for p in payload["players"] if p.get("is_self"))
        assert self_p["player_name"] == "TESTPLAYER"
        assert self_p["eliminations"] == 22
        assert self_p["damage"] == 18000
