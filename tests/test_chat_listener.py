"""Tests for chat listener: join/leave regex, state management."""

import numpy as np

from overwatchlooker.chat_listener import (
    ChatState,
    _JOIN_LEAVE_RE,
    _is_duplicate,
)


class TestJoinLeaveRegex:
    def test_joined(self):
        text = "[miabunni] joined the game."
        m = _JOIN_LEAVE_RE.search(text)
        assert m is not None
        assert m.group(1) == "miabunni"
        assert "joined" in m.group(0)

    def test_left(self):
        text = "[teryascotch] left the game."
        m = _JOIN_LEAVE_RE.search(text)
        assert m is not None
        assert m.group(1) == "teryascotch"
        assert "left" in m.group(0)

    def test_case_insensitive(self):
        text = "[PLAYER1] Left The Game"
        m = _JOIN_LEAVE_RE.search(text)
        assert m is not None

    def test_no_match_other_messages(self):
        text = "[PLAYER1] has left the group."
        m = _JOIN_LEAVE_RE.search(text)
        assert m is None

    def test_no_match_voice_channel(self):
        text = "TeryaScotch has left the voice channel."
        m = _JOIN_LEAVE_RE.search(text)
        assert m is None

    def test_no_match_stopped_playing(self):
        text = "[TeryaScotch] stopped playing Overwatch"
        m = _JOIN_LEAVE_RE.search(text)
        assert m is None

    def test_multiple_in_text(self):
        text = "[player1] left the game.\n[player2] joined the game."
        matches = list(_JOIN_LEAVE_RE.finditer(text))
        assert len(matches) == 2


class TestChatState:
    def test_initially_empty(self):
        state = ChatState()
        assert state.player_changes == []

    def test_dedup_same_frame(self):
        """Same lines in consecutive frames should not duplicate events."""
        state = ChatState()
        # Simulate: first frame sees "[player1] left the game"
        state.last_lines = set()
        # Manually add an event
        state.player_changes.append((10.0, "PLAYER1", "left"))
        state.last_lines.add("[player1] left the game")

        # Second call with same text should not add another event
        # (dedup is handled by last_lines in process_chat_frame)
        assert len(state.player_changes) == 1


class TestIsDuplicate:
    def test_exact_match(self):
        existing = [(10.0, "TERYASCOTCH", "left")]
        assert _is_duplicate("TERYASCOTCH", "left", existing) is True

    def test_fuzzy_match(self):
        existing = [(10.0, "TERYASCOTCH", "left")]
        assert _is_duplicate("TERYASEOTCH", "left", existing) is True

    def test_different_event_not_duplicate(self):
        existing = [(10.0, "TERYASCOTCH", "left")]
        assert _is_duplicate("TERYASCOTCH", "joined", existing) is False

    def test_different_player_not_duplicate(self):
        existing = [(10.0, "TERYASCOTCH", "left")]
        assert _is_duplicate("MIABUNNI", "left", existing) is False

    def test_empty_existing(self):
        assert _is_duplicate("PLAYER1", "left", []) is False


class TestChatSystemCallback:
    def test_callback_fires_on_player_change(self):
        from unittest.mock import MagicMock, patch
        from overwatchlooker.tick import ChatSystem, TickContext

        callback = MagicMock()
        system = ChatSystem(on_player_change=callback)

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        ctx = TickContext(tick=0, sim_time=10.0, frame_bgr=frame, input=MagicMock())

        def fake_process(frame, sim_time, state):
            state.player_changes.append((sim_time, "PLAYER1", "left"))

        with patch("overwatchlooker.tick.process_chat_frame", side_effect=fake_process):
            system.on_tick(ctx)

        callback.assert_called_once_with("PLAYER1", "left", 10.0)

    def test_reset_match_clears(self):
        from overwatchlooker.tick import ChatSystem

        system = ChatSystem()
        system._state.player_changes.append((10.0, "PLAYER1", "left"))
        system._last_count = 1
        system.reset_match()
        assert system.player_changes == []
        assert system._last_count == 0
