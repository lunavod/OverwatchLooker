"""Tests for tray App: hero crop lifecycle, analysis flow."""

import threading
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

    def test_reset_match_called_on_manual_submit(self, app):
        app._analyzing = False
        app._detector = MagicMock()
        app._detector.hero_map = {}
        app._detector.hero_history = {}

        with patch.object(app, "_run_analysis"):
            app._on_submit_tab("DEFEAT")

        app._detector.reset_match.assert_called_once()

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
