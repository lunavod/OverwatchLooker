"""WebSocket server for broadcasting real-time events to companion apps."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import websockets
from websockets.asyncio.server import Server, ServerConnection


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalars that sneak into event dicts."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

_logger = logging.getLogger("overwatchlooker")

# Default port for the WebSocket server
DEFAULT_PORT = 42685

# Valid commands that clients can send
COMMANDS = {"start_listening", "stop_listening", "toggle_recording",
            "submit_win", "submit_loss", "quit"}


class EventBus:
    """Thread-safe bridge between sync app code and async WebSocket server.

    App threads call emit() to post events; the async server loop picks them
    up and broadcasts to all connected clients.  Companion apps can send
    commands back; registered handlers are invoked on the main thread.
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
        self._handlers: dict[str, Callable[[], None]] = {}

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
                self._state["valid_tabs"] = self._state.get("valid_tabs", 0) + 1
            elif t == "hero_crop":
                crops = self._state.setdefault("hero_crops", [])
                name = event.get("name")
                if name and name not in crops:
                    crops.append(name)

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def register(self, command: str, handler: Callable[[], None]) -> None:
        """Register a handler for a command from companion apps."""
        self._handlers[command] = handler

    def handle_command(self, command: str) -> dict[str, Any]:
        """Dispatch a command. Returns a response dict to send back."""
        if command not in COMMANDS:
            return {"type": "error", "command": command,
                    "message": f"Unknown command: {command}"}
        handler = self._handlers.get(command)
        if handler is None:
            return {"type": "error", "command": command,
                    "message": f"Command not available: {command}"}
        try:
            handler()
            return {"type": "ok", "command": command}
        except Exception as e:
            _logger.error(f"Command {command} failed: {e}")
            return {"type": "error", "command": command, "message": str(e)}

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
            await ws.send(json.dumps(state, cls=_NumpyEncoder))
            # Process incoming commands
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    await ws.send(json.dumps(
                        {"type": "error", "message": "Invalid JSON"}))
                    continue
                cmd = msg.get("command")
                if not cmd:
                    await ws.send(json.dumps(
                        {"type": "error", "message": "Missing 'command' field"}))
                    continue
                # Run handler in a thread to avoid blocking the async loop
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self._bus.handle_command, cmd)
                await ws.send(json.dumps(response, cls=_NumpyEncoder))
        finally:
            self._clients.discard(ws)
            _logger.info(f"Companion disconnected ({len(self._clients)} clients)")

    async def _broadcast(self, event: dict[str, Any]) -> None:
        if not self._clients:
            return
        # Serialize once
        data = json.dumps(event, cls=_NumpyEncoder)
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
