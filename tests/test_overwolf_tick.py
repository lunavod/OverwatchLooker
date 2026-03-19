"""Tests for Overwolf tick-system integration: OverwolfSystem, ReplayOverwolfSource, pre-tick hooks."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from overwatchlooker.overwolf import (
    GameStateUpdate,
    GameState,
    MatchEndEvent,
    MatchStartEvent,
    OverwolfEventQueue,
    OverwolfRecordingWriter,
)
from overwatchlooker.tick import (
    OverwolfSystem,
    ReplayOverwolfSource,
    TickContext,
    TickLoop,
)


@pytest.fixture
def queue():
    return OverwolfEventQueue()


@pytest.fixture
def dummy_ctx():
    """Minimal TickContext for testing."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    input_source = MagicMock()
    return TickContext(tick=10, sim_time=1.0, frame_bgr=frame, input=input_source)


# ---------------------------------------------------------------------------
# OverwolfSystem
# ---------------------------------------------------------------------------

class TestOverwolfSystem:
    def test_drains_queue_and_dispatches(self, queue, dummy_ctx):
        received = []
        system = OverwolfSystem(queue, on_event=received.append)
        queue.push(MatchStartEvent(timestamp=100))
        queue.push(MatchEndEvent(timestamp=200))
        system.on_tick(dummy_ctx)
        assert len(received) == 2
        assert isinstance(received[0], MatchStartEvent)
        assert isinstance(received[1], MatchEndEvent)

    def test_empty_queue_noop(self, queue, dummy_ctx):
        received = []
        system = OverwolfSystem(queue, on_event=received.append)
        system.on_tick(dummy_ctx)
        assert received == []

    def test_writes_to_recording(self, queue, dummy_ctx, tmp_path):
        path = tmp_path / "test.jsonl"
        writer = OverwolfRecordingWriter(path)
        system = OverwolfSystem(queue, writer=writer)
        queue.push(MatchStartEvent(timestamp=100))
        system.on_tick(dummy_ctx)
        writer.close()
        content = path.read_text()
        assert "MatchStartEvent" in content
        assert f'"frame":{dummy_ctx.tick}' in content

    def test_set_and_clear_writer(self, queue, dummy_ctx, tmp_path):
        system = OverwolfSystem(queue)
        path = tmp_path / "test.jsonl"
        writer = OverwolfRecordingWriter(path)
        system.set_writer(writer)
        queue.push(MatchStartEvent(timestamp=100))
        system.on_tick(dummy_ctx)
        system.clear_writer()
        assert path.read_text().strip() != ""

    def test_event_handler_error_doesnt_crash(self, queue, dummy_ctx):
        def bad_handler(event):
            raise RuntimeError("boom")
        system = OverwolfSystem(queue, on_event=bad_handler)
        queue.push(MatchStartEvent(timestamp=100))
        # Should not raise
        system.on_tick(dummy_ctx)


# ---------------------------------------------------------------------------
# ReplayOverwolfSource
# ---------------------------------------------------------------------------

class TestReplayOverwolfSource:
    def test_advance_pushes_events_up_to_tick(self, queue):
        events = [
            (0, MatchStartEvent(timestamp=100)),
            (5, GameStateUpdate(state=GameState.MATCH_IN_PROGRESS, timestamp=200)),
            (10, MatchEndEvent(timestamp=300)),
        ]
        src = ReplayOverwolfSource(events, queue)

        src.advance_to(0)
        assert len(queue.drain()) == 1  # frame 0

        src.advance_to(4)
        assert len(queue.drain()) == 0  # nothing between 1-4

        src.advance_to(5)
        assert len(queue.drain()) == 1  # frame 5

        src.advance_to(100)
        assert len(queue.drain()) == 1  # frame 10

    def test_advance_idempotent(self, queue):
        events = [(5, MatchStartEvent(timestamp=100))]
        src = ReplayOverwolfSource(events, queue)
        src.advance_to(5)
        src.advance_to(5)  # should not push again
        assert len(queue.drain()) == 1

    def test_empty_events(self, queue):
        src = ReplayOverwolfSource([], queue)
        src.advance_to(100)
        assert queue.drain() == []


# ---------------------------------------------------------------------------
# TickLoop.register_pre_tick
# ---------------------------------------------------------------------------

class TestTickLoopPreTick:
    def test_pre_tick_hooks_called(self):
        """Pre-tick hooks are called with the current tick before systems."""
        from overwatchlooker.tick import ReplayFrameSource

        hook_calls = []

        class FakeReader:
            """Mimics FrameReader for ReplayFrameSource."""
            def __init__(self):
                self._count = 0
            def read_next(self):
                self._count += 1
                if self._count > 3:
                    return None
                return np.zeros((10, 10, 3), dtype=np.uint8)

        class FakeInputSource:
            def advance_to(self, tick): pass
            def is_key_held(self, key): return False
            def key_just_pressed(self, key): return False
            def key_just_released(self, key): return False
            def stop(self): pass

        frame_source = ReplayFrameSource(FakeReader())
        loop = TickLoop(fps=10, frame_source=frame_source, input_source=FakeInputSource())
        loop.register_pre_tick(lambda tick: hook_calls.append(tick))
        # Need at least one system for the barrier to work
        loop.register(lambda ctx: None, every_n_ticks=1)
        loop.run()

        assert hook_calls == [0, 1, 2]
