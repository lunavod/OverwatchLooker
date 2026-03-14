"""WebSocket server for broadcasting real-time events to companion apps."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import Any

import websockets
from websockets.asyncio.server import Server, ServerConnection

_logger = logging.getLogger("overwatchlooker")

# Default port for the WebSocket server
DEFAULT_PORT = 42685


class EventBus:
    """Thread-safe bridge between sync app code and async WebSocket server.

    App threads call emit() to post events; the async server loop picks them
    up and broadcasts to all connected clients.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state: dict[str, Any] = {
            "type": "state",
            "active": False,
            "analyzing": False,
            "hero_map": {},
            "hero_history": {},
            "hero_crops": [],
            "valid_tabs": 0,
            "recording": False,
        }
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def emit(self, event: dict[str, Any]) -> None:
        """Post an event from any thread. Non-blocking."""
        # Update snapshot state for new-client sync
        self._update_state(event)

        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def _update_state(self, event: dict[str, Any]) -> None:
        """Keep a running snapshot so new clients get current state."""
        t = event.get("type")
        with self._lock:
            if t == "state":
                self._state.update(event)
            elif t == "hero_switch":
                hm = self._state.setdefault("hero_map", {})
                hm[event["player"]] = event["hero"]
            elif t == "detection":
                self._state["last_detection"] = event.get("result")
            elif t == "analysis":
                self._state["analyzing"] = False
                self._state["last_analysis"] = event.get("data")
            elif t == "tab_capture":
                self._state["valid_tabs"] = event.get("count", 0)
            elif t == "hero_crop":
                crops = self._state.setdefault("hero_crops", [])
                name = event.get("name")
                if name and name not in crops:
                    crops.append(name)

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    async def consume(self) -> dict[str, Any]:
        return await self._queue.get()


class WsServer:
    """Async WebSocket server running in its own thread."""

    def __init__(self, bus: EventBus, port: int = DEFAULT_PORT,
                 host: str = "127.0.0.1") -> None:
        self._bus = bus
        self._port = port
        self._host = host
        self._clients: set[ServerConnection] = set()
        self._thread: threading.Thread | None = None
        self._server: Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.actual_port: int | None = None  # set after bind (useful when port=0)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._bus.set_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except RuntimeError:
            pass  # loop stopped during shutdown — expected

    async def _serve(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
        )
        # Discover actual port (useful when port=0)
        for sock in self._server.sockets:
            addr = sock.getsockname()
            self.actual_port = addr[1]
            break
        _logger.info(f"WebSocket server listening on ws://{self._host}:{self.actual_port}")

        # Broadcast loop: pull from bus, fan-out to clients
        try:
            while True:
                event = await self._bus.consume()
                await self._broadcast(event)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            if self._server is not None:
                try:
                    self._server.close()
                    await self._server.wait_closed()
                except RuntimeError:
                    pass  # loop closed during shutdown

    async def _handler(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        _logger.info(f"Companion connected ({len(self._clients)} clients)")
        try:
            # Send current state snapshot on connect
            state = self._bus.get_state()
            await ws.send(json.dumps(state))
            # Keep connection alive; ignore incoming messages
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)
            _logger.info(f"Companion disconnected ({len(self._clients)} clients)")

    async def _broadcast(self, event: dict[str, Any]) -> None:
        if not self._clients:
            return
        # Serialize once
        data = json.dumps(event)
        dead: list[ServerConnection] = []
        for ws in self._clients:
            try:
                await ws.send(data)
            except websockets.ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def stop(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)


def encode_png(png_bytes: bytes) -> str:
    """Encode PNG bytes to base64 for JSON transport."""
    return base64.b64encode(png_bytes).decode("ascii")
