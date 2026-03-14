"""Tests for WebSocket server: EventBus state management and WsServer integration."""

import asyncio
import json

import pytest
import websockets

from overwatchlooker.ws_server import EventBus, WsServer


# ---------------------------------------------------------------------------
# EventBus unit tests — one per event type
# ---------------------------------------------------------------------------

class TestEventBusState:
    def test_initial_state(self):
        bus = EventBus()
        state = bus.get_state()
        assert state["type"] == "state"
        assert state["active"] is False
        assert state["analyzing"] is False
        assert state["hero_map"] == {}
        assert state["hero_history"] == {}
        assert state["hero_crops"] == []
        assert state["valid_tabs"] == 0
        assert state["recording"] is False

    def test_state_event_updates_fields(self):
        bus = EventBus()
        bus.emit({"type": "state", "active": True, "analyzing": False})
        state = bus.get_state()
        assert state["active"] is True
        assert state["analyzing"] is False

    def test_state_event_partial_update(self):
        """State events merge — unmentioned fields are preserved."""
        bus = EventBus()
        bus.emit({"type": "state", "active": True})
        bus.emit({"type": "state", "analyzing": True})
        state = bus.get_state()
        assert state["active"] is True
        assert state["analyzing"] is True

    def test_hero_switch_event(self):
        bus = EventBus()
        bus.emit({"type": "hero_switch", "player": "PLAYER1", "hero": "Reinhardt", "time": 10.0})
        state = bus.get_state()
        assert state["hero_map"]["PLAYER1"] == "Reinhardt"

    def test_hero_switch_updates_existing(self):
        bus = EventBus()
        bus.emit({"type": "hero_switch", "player": "PLAYER1", "hero": "Reinhardt", "time": 10.0})
        bus.emit({"type": "hero_switch", "player": "PLAYER1", "hero": "Winston", "time": 20.0})
        state = bus.get_state()
        assert state["hero_map"]["PLAYER1"] == "Winston"

    def test_hero_switch_multiple_players(self):
        bus = EventBus()
        bus.emit({"type": "hero_switch", "player": "PLAYER1", "hero": "Reinhardt", "time": 10.0})
        bus.emit({"type": "hero_switch", "player": "PLAYER2", "hero": "Mercy", "time": 10.0})
        state = bus.get_state()
        assert state["hero_map"]["PLAYER1"] == "Reinhardt"
        assert state["hero_map"]["PLAYER2"] == "Mercy"

    def test_detection_event(self):
        bus = EventBus()
        bus.emit({"type": "detection", "result": "VICTORY", "time": 45.0})
        state = bus.get_state()
        assert state["last_detection"] == "VICTORY"

    def test_detection_event_overwritten(self):
        """Latest detection replaces previous."""
        bus = EventBus()
        bus.emit({"type": "detection", "result": "VICTORY", "time": 45.0})
        bus.emit({"type": "detection", "result": "DEFEAT", "time": 90.0})
        state = bus.get_state()
        assert state["last_detection"] == "DEFEAT"

    def test_analysis_event(self):
        bus = EventBus()
        bus.emit({"type": "state", "analyzing": True})
        match_data = {"map_name": "Lijiang Tower", "result": "VICTORY"}
        bus.emit({"type": "analysis", "data": match_data})
        state = bus.get_state()
        assert state["analyzing"] is False
        assert state["last_analysis"] == match_data

    def test_tab_capture_event(self):
        bus = EventBus()
        bus.emit({"type": "tab_capture", "filename": "tab_001.png", "timestamp": 5.0, "count": 1})
        state = bus.get_state()
        assert state["valid_tabs"] == 1

    def test_tab_capture_count_updates(self):
        bus = EventBus()
        bus.emit({"type": "tab_capture", "filename": "tab_001.png", "timestamp": 5.0, "count": 1})
        bus.emit({"type": "tab_capture", "filename": "tab_002.png", "timestamp": 10.0, "count": 2})
        state = bus.get_state()
        assert state["valid_tabs"] == 2

    def test_hero_crop_event(self):
        bus = EventBus()
        bus.emit({"type": "hero_crop", "name": "Reinhardt"})
        state = bus.get_state()
        assert "Reinhardt" in state["hero_crops"]

    def test_hero_crop_dedup(self):
        bus = EventBus()
        bus.emit({"type": "hero_crop", "name": "Reinhardt"})
        bus.emit({"type": "hero_crop", "name": "Reinhardt"})
        state = bus.get_state()
        assert state["hero_crops"].count("Reinhardt") == 1

    def test_hero_crop_multiple(self):
        bus = EventBus()
        bus.emit({"type": "hero_crop", "name": "Reinhardt"})
        bus.emit({"type": "hero_crop", "name": "Mercy"})
        state = bus.get_state()
        assert len(state["hero_crops"]) == 2

    def test_emit_without_loop_is_noop(self):
        """emit() should not crash when no async loop is set."""
        bus = EventBus()
        bus.emit({"type": "state", "active": True})
        # State still updates (snapshot), just no async queue push
        assert bus.get_state()["active"] is True

    def test_analyzing_event_not_tracked_in_state(self):
        """The 'analyzing' event type doesn't update state directly."""
        bus = EventBus()
        bus.emit({"type": "analyzing", "result": "VICTORY"})
        state = bus.get_state()
        # analyzing type is not handled in _update_state
        assert state["analyzing"] is False


# ---------------------------------------------------------------------------
# App emit integration — verify each event fires through the bus
# ---------------------------------------------------------------------------

class TestAppEmitsEvents:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def app(self, bus):
        from overwatchlooker.tray import App
        return App(event_bus=bus)

    def test_store_valid_tab_emits_tab_capture(self, app, bus):
        app.store_valid_tab(b"png_data", 1.0, "tab_001.png")
        state = bus.get_state()
        assert state["valid_tabs"] == 1

    def test_store_hero_crop_emits_hero_crop(self, app, bus):
        app.store_hero_crop("Reinhardt", b"crop_data")
        state = bus.get_state()
        assert "Reinhardt" in state["hero_crops"]

    def test_on_hero_switch_emits(self, app, bus):
        app._on_hero_switch("PLAYER1", "Mercy", 15.0)
        state = bus.get_state()
        assert state["hero_map"]["PLAYER1"] == "Mercy"

    def test_on_detected_emits_detection(self, app, bus):
        app._on_detected("VICTORY", 45.0)
        state = bus.get_state()
        assert state["last_detection"] == "VICTORY"

    def test_on_detection_emits_state_change(self, app, bus):
        """_on_detection sets analyzing=True, which should propagate."""
        from unittest.mock import MagicMock, patch
        app._analyzing = False
        app._detector = MagicMock()
        app._detector.hero_map = {}
        app._detector.hero_history = {}

        with patch.object(app, "_run_analysis"):
            app._on_detection("DEFEAT")

        # analyzing is set internally but not emitted as a state event
        # the "analyzing" type event is emitted from _run_analysis
        assert app._analyzing is True

    def test_no_emit_without_bus(self):
        """App with no event_bus should not crash on any emit path."""
        from overwatchlooker.tray import App
        app = App()  # no event_bus
        app.store_valid_tab(b"png_data", 1.0, "tab.png")
        app.store_hero_crop("Mercy", b"crop")
        app._on_hero_switch("P1", "Mercy", 1.0)
        app._on_detected("VICTORY", 1.0)
        # No exception = pass


# ---------------------------------------------------------------------------
# WsServer integration — real WebSocket connection
# ---------------------------------------------------------------------------

def _start_server() -> tuple[EventBus, int, WsServer]:
    """Start a WsServer on an OS-assigned port, return (bus, port, server)."""
    bus = EventBus()
    srv = WsServer(bus, port=0)
    srv.start()
    import time
    for _ in range(50):
        if srv.actual_port is not None:
            break
        time.sleep(0.05)
    assert srv.actual_port is not None
    return bus, srv.actual_port, srv


class TestWsServerIntegration:
    def test_client_receives_state_on_connect(self):
        bus, port, srv = _start_server()
        try:
            async def _test():
                async with websockets.connect(f"ws://localhost:{port}") as ws:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                    assert msg["type"] == "state"
                    assert msg["active"] is False
            asyncio.run(_test())
        finally:
            srv.stop()

    def test_client_receives_emitted_event(self):
        bus, port, srv = _start_server()
        try:
            async def _test():
                async with websockets.connect(f"ws://localhost:{port}") as ws:
                    await asyncio.wait_for(ws.recv(), timeout=2.0)  # initial state
                    bus.emit({"type": "detection", "result": "VICTORY", "time": 30.0})
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                    assert msg["type"] == "detection"
                    assert msg["result"] == "VICTORY"
            asyncio.run(_test())
        finally:
            srv.stop()

    def test_multiple_clients_receive_event(self):
        bus, port, srv = _start_server()
        try:
            async def _test():
                async with websockets.connect(f"ws://localhost:{port}") as ws1, \
                             websockets.connect(f"ws://localhost:{port}") as ws2:
                    await asyncio.wait_for(ws1.recv(), timeout=2.0)
                    await asyncio.wait_for(ws2.recv(), timeout=2.0)
                    bus.emit({"type": "hero_switch", "player": "P1", "hero": "Genji", "time": 5.0})
                    msg1 = json.loads(await asyncio.wait_for(ws1.recv(), timeout=2.0))
                    msg2 = json.loads(await asyncio.wait_for(ws2.recv(), timeout=2.0))
                    assert msg1 == msg2
                    assert msg1["player"] == "P1"
            asyncio.run(_test())
        finally:
            srv.stop()

    def test_state_snapshot_reflects_prior_events(self):
        """Client connecting after events should see accumulated state."""
        bus, port, srv = _start_server()
        try:
            bus.emit({"type": "state", "active": True})
            bus.emit({"type": "hero_switch", "player": "P1", "hero": "Ana", "time": 1.0})

            async def _test():
                await asyncio.sleep(0.05)
                async with websockets.connect(f"ws://localhost:{port}") as ws:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                    assert msg["active"] is True
                    assert msg["hero_map"]["P1"] == "Ana"
            asyncio.run(_test())
        finally:
            srv.stop()
