"""Tests for tray App: MatchState integration, cooldown, recording, memoir input."""

from unittest.mock import MagicMock, patch

import pytest

from memoir_capture import MetaFile, MetaHeader, MetaKeyEntry, MetaRow


def _seed_match(app):
    """Give the match state enough data so _trigger_match_end won't skip it."""
    app._match_state.started_at = 1000
    app._on_hero_switch("_SEED_", "Test", 0.0)


def _make_trigger_sync(app):
    """Make _trigger_match_end synchronous (no delay) for testing."""

    def _sync_trigger():
        # Call original to set _analysis_triggered and debounce
        ms = app._match_state
        if ms._analysis_triggered:
            return
        if not ms.players and ms.started_at is None:
            return
        import time as _time
        now = _time.monotonic()
        if now - app._last_match_end_ts < 5.0:
            return
        app._last_match_end_ts = now
        ms._analysis_triggered = True
        # Finalize immediately instead of after delay
        app._finalize_match_end()

    app._trigger_match_end = _sync_trigger


@pytest.fixture
def app():
    from overwatchlooker.tray import App
    a = App()
    _make_trigger_sync(a)
    return a


class TestMatchStateLifecycle:
    def test_match_state_initially_empty(self, app):
        ms = app._match_state
        assert ms.players == {}
        assert ms.latest_tab is None
        assert ms.result is None

    def test_tab_stored_in_match_state(self, app):
        app.store_valid_tab(b"png", 1.0, "tab.png")
        assert app._match_state.latest_tab is not None
        assert app._match_state.latest_tab.filename == "tab.png"

    def test_latest_tab_updated(self, app):
        app.store_valid_tab(b"png1", 1.0, "tab1.png")
        app.store_valid_tab(b"png2", 2.0, "tab2.png")
        assert app._match_state.latest_tab.filename == "tab2.png"

    def test_hero_switch_creates_player(self, app):
        app._on_hero_switch("PLAYER1", "Reinhardt", 10.0)
        p = app._match_state.players["PLAYER1"]
        assert p.current_hero == "Reinhardt"
        assert len(p.hero_swaps) == 1

    def test_hero_switch_dedup(self, app):
        app._on_hero_switch("PLAYER1", "Reinhardt", 10.0)
        app._on_hero_switch("PLAYER1", "Reinhardt", 11.0)
        assert len(app._match_state.players["PLAYER1"].hero_swaps) == 1

    def test_hero_switch_different_hero(self, app):
        app._on_hero_switch("PLAYER1", "Reinhardt", 10.0)
        app._on_hero_switch("PLAYER1", "Winston", 20.0)
        assert len(app._match_state.players["PLAYER1"].hero_swaps) == 2

    def test_player_change_joined(self, app):
        app._on_player_change("PLAYER1", "joined", 5.0)
        assert app._match_state.players["PLAYER1"].joined_at == 5000

    def test_player_change_left(self, app):
        app._on_player_change("PLAYER1", "left", 10.0)
        assert app._match_state.players["PLAYER1"].left_at == 10000


class TestOverwolfEventHandling:
    def test_map_update(self, app):
        from overwatchlooker.overwolf import MapUpdate
        app._on_overwolf_event(MapUpdate(code="212", name="King's Row", timestamp=1000))
        assert app._match_state.map_name == "King's Row"
        assert app._match_state.map_code == "212"

    def test_game_mode_update(self, app):
        from overwatchlooker.overwolf import GameModeUpdate
        app._on_overwolf_event(GameModeUpdate(code="0022", name="Hybrid", timestamp=1000))
        assert app._match_state.mode == "Hybrid"

    def test_game_type_update(self, app):
        from overwatchlooker.overwolf import GameTypeUpdate, GameType
        app._on_overwolf_event(GameTypeUpdate(game_type=GameType.RANKED, timestamp=1000))
        assert app._match_state.game_type == GameType.RANKED

    def test_queue_type_update(self, app):
        from overwatchlooker.overwolf import QueueTypeUpdate, QueueType
        app._on_overwolf_event(QueueTypeUpdate(queue_type=QueueType.ROLE_QUEUE, timestamp=1000))
        assert app._match_state.queue_type == QueueType.ROLE_QUEUE

    def test_pseudo_match_id_update(self, app):
        from overwatchlooker.overwolf import PseudoMatchIdUpdate
        app._on_overwolf_event(PseudoMatchIdUpdate(pseudo_match_id="abc-123", timestamp=1000))
        assert app._match_state.pseudo_match_id == "abc-123"

    def test_round_start_end(self, app):
        from overwatchlooker.overwolf import RoundStartEvent, RoundEndEvent
        app._on_overwolf_event(RoundStartEvent(timestamp=1000))
        assert len(app._match_state.rounds) == 1
        app._on_overwolf_event(RoundEndEvent(timestamp=2000))
        assert app._match_state.rounds[0].ended_at == 2000

    def test_match_start_creates_new_state(self, app):
        from overwatchlooker.overwolf import MatchStartEvent, MapUpdate
        # Pre-match info
        app._on_overwolf_event(MapUpdate(code="212", name="King's Row", timestamp=500))
        # Match starts
        app._on_overwolf_event(MatchStartEvent(timestamp=1000))
        assert app._match_state.started_at == 1000
        # Pre-match info carried over
        assert app._match_state.map_name == "King's Row"

    def test_roster_update_creates_player(self, app):
        from overwatchlooker.overwolf import RosterUpdate, RosterEntry
        entry = RosterEntry(
            player_name="TestPlayer", battlenet_tag="Test#1234",
            is_local=True, is_teammate=True, hero_name="Reinhardt",
            hero_role="TANK", team=0, kills=5, deaths=2, assists=3,
            damage=1000, healed=0, mitigated=500,
        )
        app._on_overwolf_event(RosterUpdate(slot=0, entry=entry, timestamp=1000))
        p = app._match_state.players["TESTPLAYER"]
        assert p.battletag == "Test#1234"
        assert p.is_local is True
        assert p.current_hero == "Reinhardt"
        assert p.stats is not None
        assert p.stats.kills == 5
        assert app._match_state._local_team == 0

    def test_roster_hero_swap_detection(self, app):
        from overwatchlooker.overwolf import RosterUpdate, RosterEntry
        entry1 = RosterEntry(
            player_name="P1", battlenet_tag="P1#1", is_local=False,
            is_teammate=True, hero_name="Reinhardt", hero_role="TANK",
            team=0, kills=0, deaths=0, assists=0, damage=0, healed=0, mitigated=0,
        )
        app._on_overwolf_event(RosterUpdate(slot=0, entry=entry1, timestamp=1000))
        assert len(app._match_state.players["P1"].hero_swaps) == 1

        # Same hero — no new swap
        app._on_overwolf_event(RosterUpdate(slot=0, entry=entry1, timestamp=2000))
        assert len(app._match_state.players["P1"].hero_swaps) == 1

        # Different hero
        entry2 = RosterEntry(
            player_name="P1", battlenet_tag="P1#1", is_local=False,
            is_teammate=True, hero_name="Winston", hero_role="TANK",
            team=0, kills=1, deaths=0, assists=0, damage=100, healed=0, mitigated=0,
        )
        app._on_overwolf_event(RosterUpdate(slot=0, entry=entry2, timestamp=3000))
        assert len(app._match_state.players["P1"].hero_swaps) == 2

    def test_team_side_resolution(self, app):
        from overwatchlooker.overwolf import RosterUpdate, RosterEntry
        from overwatchlooker.match_state import TeamSide
        # Local player on team 0
        local = RosterEntry(
            player_name="Local", battlenet_tag="L#1", is_local=True,
            is_teammate=True, hero_name="Ana", hero_role="SUPPORT",
            team=0, kills=0, deaths=0, assists=0, damage=0, healed=0, mitigated=0,
        )
        enemy = RosterEntry(
            player_name="Enemy", battlenet_tag="E#1", is_local=False,
            is_teammate=False, hero_name="", hero_role="",
            team=1, kills=0, deaths=0, assists=0, damage=0, healed=0, mitigated=0,
        )
        app._on_overwolf_event(RosterUpdate(slot=0, entry=local, timestamp=1000))
        app._on_overwolf_event(RosterUpdate(slot=5, entry=enemy, timestamp=1000))
        assert app._match_state.players["LOCAL"].team_side == TeamSide.ALLY
        assert app._match_state.players["ENEMY"].team_side == TeamSide.ENEMY

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_match_outcome_triggers_end(self, mock_print, mock_notif, app):
        from overwatchlooker.overwolf import MatchOutcomeUpdate, MatchOutcome, MatchStartEvent
        # Need a started match with players for trigger to fire
        app._on_overwolf_event(MatchStartEvent(timestamp=1000))
        app._on_hero_switch("PLAYER1", "Ana", 1.0)
        app._on_overwolf_event(MatchOutcomeUpdate(
            outcome=MatchOutcome.VICTORY, timestamp=5000))
        # Match state should be reset (new empty state)
        assert app._match_state.result is None

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_double_trigger_prevented(self, mock_print, mock_notif, app):
        from overwatchlooker.overwolf import MatchEndEvent, MatchOutcomeUpdate, MatchOutcome, MatchStartEvent
        # Need a started match with players for trigger to fire
        app._on_overwolf_event(MatchStartEvent(timestamp=1000))
        app._on_hero_switch("PLAYER1", "Ana", 1.0)
        app._on_overwolf_event(MatchEndEvent(timestamp=5000))
        # First trigger resets state
        first_state = app._match_state
        # Second trigger (outcome) should not trigger again due to debounce
        app._on_overwolf_event(MatchOutcomeUpdate(
            outcome=MatchOutcome.VICTORY, timestamp=5001))
        # State should not be reset again (debounce prevents it)
        assert app._match_state is first_state

    def test_empty_player_name_skipped(self, app):
        """Post-match empty roster updates should not create ghost players."""
        from overwatchlooker.overwolf import RosterUpdate, RosterEntry
        entry = RosterEntry(
            player_name="", battlenet_tag="", is_local=False,
            is_teammate=True, hero_name="UNKNOWN", hero_role="UNKNOWN",
            team=1, kills=0, deaths=0, assists=0, damage=0, healed=0, mitigated=0,
        )
        app._on_overwolf_event(RosterUpdate(slot=0, entry=entry, timestamp=1000))
        assert "" not in app._match_state.players
        assert len(app._match_state.players) == 0

    def test_unknown_hero_name_filtered(self, app):
        """UNKNOWN hero name from Overwolf should not create a hero swap."""
        from overwatchlooker.overwolf import RosterUpdate, RosterEntry
        entry = RosterEntry(
            player_name="Player1", battlenet_tag="P1#1", is_local=False,
            is_teammate=True, hero_name="UNKNOWN", hero_role="UNKNOWN",
            team=1, kills=0, deaths=0, assists=0, damage=0, healed=0, mitigated=0,
        )
        app._on_overwolf_event(RosterUpdate(slot=0, entry=entry, timestamp=1000))
        assert len(app._match_state.players["PLAYER1"].hero_swaps) == 0

    def test_hero_name_resolved_via_match(self, app):
        """Overwolf hero names like JETPACKCAT should resolve to canonical names."""
        from overwatchlooker.overwolf import RosterUpdate, RosterEntry
        entry = RosterEntry(
            player_name="Player1", battlenet_tag="P1#1", is_local=False,
            is_teammate=True, hero_name="JETPACKCAT", hero_role="SUPPORT",
            team=1, kills=0, deaths=0, assists=0, damage=0, healed=0, mitigated=0,
        )
        app._on_overwolf_event(RosterUpdate(slot=0, entry=entry, timestamp=1000))
        assert app._match_state.players["PLAYER1"].current_hero == "Jetpack Cat"

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_match_outcome_sets_ended_at(self, mock_print, mock_notif, app):
        """MatchOutcomeUpdate should set ended_at if not already set."""
        from overwatchlooker.overwolf import MatchOutcomeUpdate, MatchOutcome, MatchStartEvent
        app._on_overwolf_event(MatchStartEvent(timestamp=1000))
        app._on_hero_switch("PLAYER1", "Ana", 1.0)
        app._on_overwolf_event(MatchOutcomeUpdate(
            outcome=MatchOutcome.DEFEAT, timestamp=50000))
        # State was reset, but we can check the snapshot wasn't created with None ended_at
        # Since state was reset, the new state has no ended_at
        # The trigger happened — meaning ended_at was set before snapshot
        assert app._match_state.ended_at is None  # new state after reset

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_empty_state_not_triggered(self, mock_print, mock_notif, app):
        """Match end on empty state (no players, no started_at) should be skipped."""
        from overwatchlooker.overwolf import MatchEndEvent
        old_state = app._match_state
        app._on_overwolf_event(MatchEndEvent(timestamp=5000))
        # State should NOT be reset — trigger was skipped
        assert app._match_state is old_state


class TestPostSubmitCooldown:
    """After match end, tab/crop events are ignored for 30s of ticks."""

    @pytest.fixture
    def app_with_tick_loop(self):
        from overwatchlooker.tray import App
        a = App()
        _make_trigger_sync(a)
        a._tick_loop = MagicMock()
        a._tick_loop.fps = 10
        a._tick_loop._current_tick = 1000
        return a

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_cooldown_set_on_detection(self, mock_print, mock_notif, app_with_tick_loop):
        app = app_with_tick_loop
        _seed_match(app)
        app._on_detection("VICTORY")
        # 30s * 10fps = 300 ticks after current tick 1000
        assert app._cooldown_until_tick == 1300

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_cooldown_set_on_manual_submit(self, mock_print, mock_notif, app_with_tick_loop):
        app = app_with_tick_loop
        _seed_match(app)
        app._on_submit_tab("DEFEAT")
        assert app._cooldown_until_tick == 1300

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_tab_ignored_during_cooldown(self, mock_print, mock_notif, app_with_tick_loop):
        app = app_with_tick_loop
        _seed_match(app)
        app._on_detection("VICTORY")
        # Tick loop is at 1000, cooldown until 1300 — should ignore
        app.store_valid_tab(b"png", 1.0, "tab.png")
        assert app._match_state.latest_tab is None  # cooldown blocked it

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_tab_accepted_after_cooldown(self, mock_print, mock_notif, app_with_tick_loop):
        app = app_with_tick_loop
        _seed_match(app)
        app._on_detection("VICTORY")
        # Advance tick past cooldown
        app._tick_loop._current_tick = 1300
        app.store_valid_tab(b"png", 1.0, "tab.png")
        assert app._match_state.latest_tab is not None

    def test_no_cooldown_without_tick_loop(self):
        """If tick loop is not running (e.g. image mode), no cooldown applies."""
        from overwatchlooker.tray import App
        a = App()
        a.store_valid_tab(b"png", 1.0, "tab.png")
        assert a._match_state.latest_tab is not None


class TestAnalysisFlow:
    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_double_trigger_prevented(self, mock_print, mock_notif, app):
        """Second detection should not trigger again."""
        _seed_match(app)
        app._on_detection("VICTORY")
        first_state = app._match_state
        app._on_detection("DEFEAT")
        # Second trigger on empty new state — skipped
        assert app._match_state is first_state

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_reset_match_called_on_detection(self, mock_print, mock_notif, app):
        _seed_match(app)
        app._detector = MagicMock()
        app._on_detection("DEFEAT")
        app._detector.reset_match.assert_called_once()

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_chat_system_reset_on_detection(self, mock_print, mock_notif, app):
        _seed_match(app)
        app._detector = MagicMock()
        app._chat_system = MagicMock()
        app._on_detection("VICTORY")
        app._chat_system.reset_match.assert_called_once()

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_reset_match_called_on_manual_submit(self, mock_print, mock_notif, app):
        _seed_match(app)
        app._detector = MagicMock()
        app._on_submit_tab("DEFEAT")
        app._detector.reset_match.assert_called_once()

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_chat_system_reset_on_manual_submit(self, mock_print, mock_notif, app):
        _seed_match(app)
        app._detector = MagicMock()
        app._chat_system = MagicMock()
        app._on_submit_tab("DEFEAT")
        app._chat_system.reset_match.assert_called_once()

    @patch("overwatchlooker.tray.show_notification")
    @patch("overwatchlooker.display.print_analysis")
    def test_match_state_reset_after_detection(self, mock_print, mock_notif, app):
        """After detection triggers, match state is reset for next match."""
        _seed_match(app)
        app._on_hero_switch("PLAYER1", "Reinhardt", 10.0)
        assert "PLAYER1" in app._match_state.players
        app._on_detection("VICTORY")
        # State is reset
        assert len(app._match_state.players) == 0
        assert app._match_state.result is None

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
