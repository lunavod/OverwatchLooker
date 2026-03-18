"""Tests for tray App: hero crop lifecycle, analysis flow, memoir input."""

from unittest.mock import MagicMock, patch

import pytest

from memoir_capture import MetaFile, MetaHeader, MetaKeyEntry, MetaRow


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

    def test_recording_start_emits_ws_state(self, app):
        app._bus = MagicMock()
        mock_engine = MagicMock()
        mock_engine.start_recording.return_value = MagicMock(video_path="recordings/test/recording.mp4")
        app._engine = mock_engine
        with patch("overwatchlooker.tray.show_notification"), \
             patch("overwatchlooker.tray._RECORDINGS_DIR"):
            app._on_toggle_recording(None, None)
        app._bus.emit.assert_called_with({"type": "state", "recording": True})

    def test_recording_stop_emits_ws_state(self, app):
        app._bus = MagicMock()
        mock_engine = MagicMock()
        app._engine = mock_engine
        app._recording = True
        with patch("overwatchlooker.tray.show_notification"):
            app._on_toggle_recording(None, None)
        app._bus.emit.assert_called_with({"type": "state", "recording": False})

class TestMemoirInputSource:
    """Tests for MemoirInputSource keyboard bitmask decoding."""

    def _make_source(self, key_table=None):
        from overwatchlooker.tick import MemoirInputSource
        frame_source = MagicMock()
        frame_source._last_keyboard_mask = 0
        if key_table is None:
            key_table = [
                {"bit_index": 0, "name": "tab"},
                {"bit_index": 1, "name": "alt_l"},
                {"bit_index": 2, "name": "alt_r"},
            ]
        return MemoirInputSource(frame_source, key_table), frame_source

    def test_no_keys_held_initially(self):
        src, _ = self._make_source()
        src.advance_to(0)
        assert not src.is_key_held("tab")
        assert not src.is_key_held("alt_l")

    def test_tab_pressed(self):
        src, fs = self._make_source()
        fs._last_keyboard_mask = 0b001  # bit 0 = tab
        src.advance_to(0)
        assert src.is_key_held("tab")
        assert src.key_just_pressed("tab")
        assert not src.is_key_held("alt_l")

    def test_tab_held_then_released(self):
        src, fs = self._make_source()
        fs._last_keyboard_mask = 0b001
        src.advance_to(0)
        assert src.is_key_held("tab")

        fs._last_keyboard_mask = 0b000
        src.advance_to(1)
        assert not src.is_key_held("tab")
        assert src.key_just_released("tab")
        assert not src.key_just_pressed("tab")

    def test_multiple_keys(self):
        src, fs = self._make_source()
        fs._last_keyboard_mask = 0b011  # tab + alt_l
        src.advance_to(0)
        assert src.is_key_held("tab")
        assert src.is_key_held("alt_l")
        assert not src.is_key_held("alt_r")

    def test_held_key_not_re_pressed(self):
        src, fs = self._make_source()
        fs._last_keyboard_mask = 0b001
        src.advance_to(0)
        assert src.key_just_pressed("tab")

        # Same mask next tick — held but not just pressed
        src.advance_to(1)
        assert src.is_key_held("tab")
        assert not src.key_just_pressed("tab")


class TestSynthesizeEvents:
    """Tests for replay _synthesize_events from .meta keyboard masks."""

    def _make_meta(self, keys, rows):
        header = MetaHeader(magic=b"RCMETA1\x00", version=1,
                            created_unix_ns=0, key_count=len(keys))
        return MetaFile(header=header, keys=keys, rows=rows)

    def _make_row(self, frame_index, mask):
        return MetaRow(frame_id=frame_index, record_frame_index=frame_index,
                       capture_qpc=0, host_accept_qpc=0, keyboard_mask=mask,
                       width=1920, height=1080, analysis_stride=1)

    def test_key_down_on_bit_set(self):
        from overwatchlooker.recording.replay import _synthesize_events
        keys = [MetaKeyEntry(bit_index=0, virtual_key=0x09, name="tab")]
        rows = [self._make_row(0, 0b0), self._make_row(1, 0b1)]
        events = _synthesize_events(self._make_meta(keys, rows))
        assert len(events) == 1
        assert events[0] == {"frame": 1, "type": "key_down", "key": "tab"}

    def test_key_up_on_bit_clear(self):
        from overwatchlooker.recording.replay import _synthesize_events
        keys = [MetaKeyEntry(bit_index=0, virtual_key=0x09, name="tab")]
        rows = [self._make_row(0, 0b0), self._make_row(1, 0b1), self._make_row(2, 0b0)]
        events = _synthesize_events(self._make_meta(keys, rows))
        assert len(events) == 2
        assert events[0] == {"frame": 1, "type": "key_down", "key": "tab"}
        assert events[1] == {"frame": 2, "type": "key_up", "key": "tab"}

    def test_multiple_keys_simultaneous(self):
        from overwatchlooker.recording.replay import _synthesize_events
        keys = [
            MetaKeyEntry(bit_index=0, virtual_key=0x09, name="tab"),
            MetaKeyEntry(bit_index=1, virtual_key=0xA4, name="alt_l"),
        ]
        rows = [self._make_row(0, 0b00), self._make_row(1, 0b11)]
        events = _synthesize_events(self._make_meta(keys, rows))
        assert len(events) == 2
        names = {e["key"] for e in events}
        assert names == {"tab", "alt_l"}

    def test_no_events_when_mask_unchanged(self):
        from overwatchlooker.recording.replay import _synthesize_events
        keys = [MetaKeyEntry(bit_index=0, virtual_key=0x09, name="tab")]
        # Both frames have mask=0 — no transitions
        rows = [self._make_row(0, 0b0), self._make_row(1, 0b0)]
        events = _synthesize_events(self._make_meta(keys, rows))
        assert len(events) == 0


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
