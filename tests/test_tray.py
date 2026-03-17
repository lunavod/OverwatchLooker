"""Tests for tray App: hero crop lifecycle, analysis flow."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def app():
    from overwatchlooker.tray import App
    return App()


class TestHeroCropLifecycle:
    def test_hero_crops_initially_empty(self, app):
        assert app._hero_crops == {}

    def test_hero_crop_stored(self, app):
        """Simulate storing a hero crop under lock."""
        with app._lock:
            app._hero_crops["Reinhardt"] = b"crop_data"
        assert "Reinhardt" in app._hero_crops

    def test_hero_crop_dedup(self, app):
        """Same hero name should not be stored twice."""
        from overwatchlooker.heroes import edit_distance
        name1 = "Reinhardt"
        name2 = "Reinhardt"
        with app._lock:
            app._hero_crops[name1] = b"crop1"
            if not any(edit_distance(name2.lower(), k.lower()) <= 2 for k in app._hero_crops):
                app._hero_crops[name2] = b"crop2"
        assert len(app._hero_crops) == 1

    def test_hero_crop_different_heroes(self, app):
        with app._lock:
            app._hero_crops["Reinhardt"] = b"crop1"
            app._hero_crops["Juno"] = b"crop2"
        assert len(app._hero_crops) == 2

    def test_hero_crops_cleared_on_detection(self, app):
        app._hero_crops["Reinhardt"] = b"crop"
        app._analyzing = False
        # Mock detector
        app._detector = MagicMock()
        app._detector.hero_map = {}
        app._detector.hero_history = {}

        with patch.object(app, "_run_analysis"):
            app._on_detection("VICTORY")

        assert app._hero_crops == {}

    def test_hero_crops_cleared_on_manual_submit(self, app):
        app._hero_crops["Juno"] = b"crop"
        app._analyzing = False
        app._detector = MagicMock()
        app._detector.hero_map = {}
        app._detector.hero_history = {}

        with patch.object(app, "_run_analysis"):
            app._on_submit_tab("VICTORY")

        assert app._hero_crops == {}


class TestPostSubmitCooldown:
    """After data is gathered for analysis, tab/crop events are ignored for 30s of ticks."""

    @pytest.fixture
    def app_with_tick_loop(self):
        from overwatchlooker.tray import App
        a = App()
        a._tick_loop = MagicMock()
        a._tick_loop.fps = 10
        a._tick_loop._current_tick = 1000
        a._analyzing = False
        a._detector = MagicMock()
        a._detector.hero_map = {}
        a._detector.hero_history = {}
        return a

    def test_cooldown_set_on_detection(self, app_with_tick_loop):
        app = app_with_tick_loop
        with patch.object(app, "_run_analysis"):
            app._on_detection("VICTORY")
        # 30s * 10fps = 300 ticks after current tick 1000
        assert app._cooldown_until_tick == 1300

    def test_cooldown_set_on_manual_submit(self, app_with_tick_loop):
        app = app_with_tick_loop
        with patch.object(app, "_run_analysis"):
            app._on_submit_tab("DEFEAT")
        assert app._cooldown_until_tick == 1300

    def test_tab_ignored_during_cooldown(self, app_with_tick_loop):
        app = app_with_tick_loop
        with patch.object(app, "_run_analysis"):
            app._on_detection("VICTORY")
        # Tick loop is at 1000, cooldown until 1300 — should ignore
        app.store_valid_tab(b"png", 1.0, "tab.png")
        assert len(app._valid_tabs) == 0

    def test_crop_ignored_during_cooldown(self, app_with_tick_loop):
        app = app_with_tick_loop
        with patch.object(app, "_run_analysis"):
            app._on_detection("VICTORY")
        app.store_hero_crop("Reinhardt", b"crop")
        assert app._hero_crops == {}

    def test_tab_accepted_after_cooldown(self, app_with_tick_loop):
        app = app_with_tick_loop
        with patch.object(app, "_run_analysis"):
            app._on_detection("VICTORY")
        # Advance tick past cooldown
        app._tick_loop._current_tick = 1300
        app.store_valid_tab(b"png", 1.0, "tab.png")
        assert len(app._valid_tabs) == 1

    def test_crop_accepted_after_cooldown(self, app_with_tick_loop):
        app = app_with_tick_loop
        with patch.object(app, "_run_analysis"):
            app._on_detection("VICTORY")
        app._tick_loop._current_tick = 1300
        app.store_hero_crop("Reinhardt", b"crop")
        assert "Reinhardt" in app._hero_crops

    def test_no_cooldown_without_tick_loop(self):
        """If tick loop is not running (e.g. image mode), no cooldown applies."""
        from overwatchlooker.tray import App
        a = App()
        a.store_valid_tab(b"png", 1.0, "tab.png")
        assert len(a._valid_tabs) == 1


class TestAnalysisFlow:
    def test_analyzing_lock_prevents_double(self, app):
        """Second detection while analyzing should be ignored."""
        app._analyzing = True
        app._detector = MagicMock()

        with patch.object(app, "_run_analysis") as mock_run:
            app._on_detection("VICTORY")
            mock_run.assert_not_called()

    def test_reset_match_called_on_detection(self, app):
        app._analyzing = False
        app._detector = MagicMock()
        app._detector.hero_map = {}
        app._detector.hero_history = {}

        with patch.object(app, "_run_analysis"):
            app._on_detection("DEFEAT")

        app._detector.reset_match.assert_called_once()

    def test_chat_system_reset_on_detection(self, app):
        app._analyzing = False
        app._detector = MagicMock()
        app._detector.hero_map = {}
        app._detector.hero_history = {}
        app._chat_system = MagicMock()
        app._chat_system.player_changes = []

        with patch.object(app, "_run_analysis"):
            app._on_detection("VICTORY")

        app._chat_system.reset_match.assert_called_once()

    def test_reset_match_called_on_manual_submit(self, app):
        app._analyzing = False
        app._detector = MagicMock()
        app._detector.hero_map = {}
        app._detector.hero_history = {}

        with patch.object(app, "_run_analysis"):
            app._on_submit_tab("DEFEAT")

        app._detector.reset_match.assert_called_once()

    def test_chat_system_reset_on_manual_submit(self, app):
        app._analyzing = False
        app._detector = MagicMock()
        app._detector.hero_map = {}
        app._detector.hero_history = {}
        app._chat_system = MagicMock()
        app._chat_system.player_changes = []

        with patch.object(app, "_run_analysis"):
            app._on_submit_tab("DEFEAT")

        app._chat_system.reset_match.assert_called_once()

    def test_final_hero_removed_from_crops(self):
        """The hero visible in the final screenshot should be removed from crops."""
        from overwatchlooker.heroes import edit_distance

        hero_crops = {"Juno": b"juno_crop", "Reinhardt": b"rein_crop"}
        final_hero = "Juno"

        # This is the logic from _run_analysis
        to_remove = [k for k in hero_crops
                     if edit_distance(final_hero.lower(), k.lower()) <= 2]
        for k in to_remove:
            del hero_crops[k]

        assert "Juno" not in hero_crops
        assert "Reinhardt" in hero_crops
        assert len(hero_crops) == 1
