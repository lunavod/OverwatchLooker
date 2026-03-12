"""Tests for subtitle listener: hero extraction regex, dedup, state management."""

import re
import time

import pytest

from overwatchlooker.heroes import edit_distance, match_hero_name


# The regex used in SubtitleListener._poll to extract hero switches
_HERO_REGEX = r"\[(\w+)\s+\(([^)]+)\)\]"


class TestHeroRegex:
    def test_basic_extraction(self):
        text = "[LUNAVOD (Reinhardt)]"
        matches = re.findall(_HERO_REGEX, text)
        assert len(matches) == 1
        assert matches[0] == ("LUNAVOD", "Reinhardt")

    def test_multiple_entries(self):
        text = "[PLAYER1 (Juno)] [PLAYER2 (Genji)]"
        matches = re.findall(_HERO_REGEX, text)
        assert len(matches) == 2

    def test_hero_with_spaces(self):
        text = "[PLAYER1 (Wrecking Ball)]"
        matches = re.findall(_HERO_REGEX, text)
        assert len(matches) == 1
        assert matches[0][1] == "Wrecking Ball"

    def test_athena_extracted(self):
        """Regex extracts ATHENA but the listener code skips it."""
        text = "[ATHENA (Victory)]"
        matches = re.findall(_HERO_REGEX, text)
        assert len(matches) == 1
        assert matches[0][0] == "ATHENA"

    def test_multiline(self):
        text = "[player1 (ana)]\n[player2 (mercy)]"
        matches = re.finditer(_HERO_REGEX, text)
        results = [(m.group(1).upper(), m.group(2).strip().title()) for m in matches]
        assert len(results) == 2


class TestVictoryDefeatRegex:
    """Test the regex patterns used to detect VICTORY/DEFEAT from Athena subtitles."""

    def test_victory_clean(self):
        assert re.search(r"athena\W{0,3}\s*victory", "athena victory")

    def test_defeat_clean(self):
        assert re.search(r"athena\W{0,3}\s*defeat", "athena defeat")

    def test_victory_with_brackets(self):
        assert re.search(r"athena\W{0,3}\s*victory", "[athena] victory.")

    def test_defeat_with_noise(self):
        assert re.search(r"athena\W{0,3}\s*defeat", "athena:  defeat")

    def test_no_false_positive(self):
        assert not re.search(r"athena\W{0,3}\s*victory", "player1 victory")
        assert not re.search(r"athena\W{0,3}\s*defeat", "defeat is near")


class TestSubtitleListenerState:
    """Test SubtitleListener hero tracking state without actually polling the screen."""

    @pytest.fixture
    def listener(self):
        from overwatchlooker.subtitle_listener import SubtitleListener
        return SubtitleListener(on_match=lambda r: None)

    def test_hero_map_initially_empty(self, listener):
        assert listener.hero_map == {}

    def test_hero_history_initially_empty(self, listener):
        assert listener.hero_history == {}

    def test_reset_match_clears(self, listener):
        # Manually populate internal state
        listener._hero_map["PLAYER1"] = "Reinhardt"
        listener._hero_history["PLAYER1"] = [(100.0, "Reinhardt")]
        listener.reset_match()
        assert listener.hero_map == {}
        assert listener.hero_history == {}

    def test_hero_map_property_is_copy(self, listener):
        listener._hero_map["PLAYER1"] = "Ana"
        external = listener.hero_map
        external["PLAYER1"] = "Moira"
        assert listener._hero_map["PLAYER1"] == "Ana"

    def test_hero_history_property_is_copy(self, listener):
        listener._hero_history["PLAYER1"] = [(100.0, "Ana")]
        external = listener.hero_history
        external["PLAYER1"].append((200.0, "Moira"))
        assert len(listener._hero_history["PLAYER1"]) == 1

    def test_hero_tracking_simulation(self, listener):
        """Simulate the hero extraction logic from _poll."""
        now = time.monotonic()

        # Simulate processing subtitle frames
        frames = [
            {("PLAYER1", "Reinhardt")},
            {("PLAYER1", "Reinhardt")},   # same hero, should dedup
            {("PLAYER1", "Juno")},         # new hero
            {("PLAYER2", "Ana")},
        ]

        for frame_heroes in frames:
            for username, hero in frame_heroes:
                hero = match_hero_name(hero) or hero
                history = listener._hero_history.get(username)
                if history:
                    last_hero = history[-1][1]
                    if edit_distance(hero.lower(), last_hero.lower()) <= 2:
                        continue
                if username not in listener._hero_history:
                    listener._hero_history[username] = []
                listener._hero_history[username].append((now, hero))
                listener._hero_map[username] = hero

        assert listener._hero_map["PLAYER1"] == "Juno"
        assert listener._hero_map["PLAYER2"] == "Ana"
        assert len(listener._hero_history["PLAYER1"]) == 2  # Rein + Juno (dedup'd)
        assert len(listener._hero_history["PLAYER2"]) == 1

    def test_fuzzy_dedup(self, listener):
        """OCR noise like 'Reinhardtt' should not create a new history entry."""
        listener._hero_history["P1"] = [(100.0, "Reinhardt")]
        # Simulate what _poll does
        hero = "Reinhardtt"
        last_hero = listener._hero_history["P1"][-1][1]
        should_skip = edit_distance(hero.lower(), last_hero.lower()) <= 2
        assert should_skip is True
