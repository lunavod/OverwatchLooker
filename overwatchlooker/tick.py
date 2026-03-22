"""Tick-based frame loop for both live and replay modes."""

from __future__ import annotations

import logging
import datetime
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from overwatchlooker.overwolf import OverwolfEventQueue, OverwolfRecordingWriter

import cv2
import numpy as np

from overwatchlooker.display import print_status
from overwatchlooker.screenshot import (
    is_ow2_tab_screen_bgr,
    save_screenshot,
)
from overwatchlooker.chat_listener import ChatState, process_chat_frame
from overwatchlooker.subtitle_listener import SubtitleState, process_subtitle_frame

_logger = logging.getLogger("overwatchlooker")


class InputSource(Protocol):
    def advance_to(self, tick: int) -> None: ...
    def is_key_held(self, key: str) -> bool: ...
    def key_just_pressed(self, key: str) -> bool: ...
    def key_just_released(self, key: str) -> bool: ...
    def stop(self) -> None: ...


@dataclass
class TickContext:
    tick: int
    sim_time: float
    frame_bgr: np.ndarray
    input: InputSource


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class MemoirFrameSource:
    """Captures frames via a memoir CaptureEngine, paces at real FPS."""

    def __init__(self, engine, fps: int):
        self._engine = engine
        self._fps = fps
        self._interval = 1.0 / fps
        self._last_time = 0.0
        self._last_keyboard_mask: int = 0

    def next_frame(self) -> np.ndarray | None:
        # Pace at FPS
        now = time.perf_counter()
        sleep_time = self._interval - (now - self._last_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.perf_counter()

        packet = self._engine.get_next_frame(timeout_ms=200)
        if packet is None:
            return None

        try:
            self._last_keyboard_mask = packet.keyboard_mask
            # cpu_bgra is (H, W, 4) — slice to BGR and copy before releasing
            bgr = packet.cpu_bgra[:, :, :3].copy()
        finally:
            packet.release()

        return bgr


class ReplayFrameSource:
    """Reads frames sequentially from a FrameReader, no sleeps."""

    def __init__(self, reader):
        self._reader = reader

    def next_frame(self) -> np.ndarray | None:
        return self._reader.read_next()


# ---------------------------------------------------------------------------
# Input sources
# ---------------------------------------------------------------------------

class MemoirInputSource:
    """Reads keyboard state from memoir FramePacket keyboard_mask bitmasks."""

    def __init__(self, frame_source: MemoirFrameSource, key_table: list[dict]):
        """
        Args:
            frame_source: The MemoirFrameSource to read _last_keyboard_mask from.
            key_table: List of {bit_index, name} dicts from the key map passed
                       to the CaptureEngine. Names should match what the app
                       expects (e.g. "tab", "alt_l", "alt_r").
        """
        self._frame_source = frame_source
        self._bit_to_name: dict[int, str] = {
            entry["bit_index"]: entry["name"] for entry in key_table
        }

        self._held: set[str] = set()
        self._just_pressed: set[str] = set()
        self._just_released: set[str] = set()
        self._prev_mask: int = 0

    def advance_to(self, tick: int) -> None:
        mask = self._frame_source._last_keyboard_mask
        # Decode mask into current held set
        current: set[str] = set()
        for bit, name in self._bit_to_name.items():
            if mask & (1 << bit):
                current.add(name)

        self._just_pressed = current - self._held
        self._just_released = self._held - current
        self._held = current
        self._prev_mask = mask

    def is_key_held(self, key: str) -> bool:
        return key in self._held

    def key_just_pressed(self, key: str) -> bool:
        return key in self._just_pressed

    def key_just_released(self, key: str) -> bool:
        return key in self._just_released

    def stop(self) -> None:
        pass


class ReplayInputSource:
    """Replays keyboard events synthesized from .meta keyboard_mask diffs."""

    def __init__(self, events: list[dict]):
        self._events = [e for e in events if e.get("type") in ("key_down", "key_up")]
        self._idx = 0
        self._held: set[str] = set()
        self._just_pressed: set[str] = set()
        self._just_released: set[str] = set()

    def advance_to(self, tick: int) -> None:
        self._just_pressed = set()
        self._just_released = set()

        while self._idx < len(self._events):
            ev = self._events[self._idx]
            if ev["frame"] > tick:
                break
            key = ev.get("key", "")
            if ev["type"] == "key_down":
                self._just_pressed.add(key)
                self._held.add(key)
            elif ev["type"] == "key_up":
                self._just_released.add(key)
                self._held.discard(key)
            self._idx += 1

    def is_key_held(self, key: str) -> bool:
        return key in self._held

    def key_just_pressed(self, key: str) -> bool:
        return key in self._just_pressed

    def key_just_released(self, key: str) -> bool:
        return key in self._just_released

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

class _TabState:
    """States for the TabCaptureSystem state machine."""
    IDLE = "idle"              # Tab not held, waiting for press
    INITIAL_WAIT = "wait"      # Tab pressed, waiting 1s for screen to render
    CHECK = "check"            # Ready to check this frame
    RETRY_WAIT = "retry_wait"  # Frame wasn't valid, waiting 0.5s before retry
    DONE = "done"              # Got valid capture, waiting for Tab release
    COOLDOWN = "cooldown"      # Tab released, 0.5s debounce before accepting new press


class TabCaptureSystem:
    """Captures one Tab screen screenshot per Tab press.

    State machine that mirrors the original _capture_loop timing:
      IDLE → Tab press → INITIAL_WAIT (1s) → CHECK → valid? → DONE
                                                   → invalid? → RETRY_WAIT (0.5s) → CHECK → ...
      DONE/any → Tab release → COOLDOWN (0.5s) → IDLE
    """

    def __init__(self, app, fps: int = 10):
        self._app = app
        self._state = _TabState.IDLE
        self._countdown = 0  # ticks remaining in current wait
        self._got_valid = False

        self._initial_wait = max(1, int(fps * 0.5))
        self._retry_wait = 1
        self._cooldown_wait = max(1, int(fps * 0.5))

    def on_tick(self, ctx: TickContext) -> None:
        tab_held = ctx.input.is_key_held("tab")
        alt_held = ctx.input.is_key_held("alt_l") or ctx.input.is_key_held("alt_r")
        tab_active = tab_held and not alt_held

        if self._state == _TabState.IDLE:
            if tab_active:
                self._got_valid = False
                self._state = _TabState.INITIAL_WAIT
                self._countdown = self._initial_wait

        elif self._state == _TabState.INITIAL_WAIT:
            if not tab_active:
                self._on_tab_released()
                return
            if self._countdown > 0:
                self._countdown -= 1
            else:
                self._state = _TabState.CHECK

        elif self._state == _TabState.CHECK:
            if not tab_active:
                self._on_tab_released()
                return
            is_tab, reason = is_ow2_tab_screen_bgr(ctx.frame_bgr)
            if is_tab:
                png_bytes = cv2.imencode(".png", ctx.frame_bgr)[1].tobytes()
                saved_path = save_screenshot(png_bytes, tick=ctx.tick)
                self._app.store_valid_tab(png_bytes, ctx.sim_time, saved_path.name,
                                         tick=ctx.tick)
                self._got_valid = True
                self._state = _TabState.DONE
            else:
                _logger.info(f"Tab screen rejected: {reason}")
                # Stay in CHECK — retry on next tick

        elif self._state == _TabState.RETRY_WAIT:
            if not tab_active:
                self._on_tab_released()
                return
            if self._countdown > 0:
                self._countdown -= 1
            else:
                self._state = _TabState.CHECK

        elif self._state == _TabState.DONE:
            if not tab_active:
                self._on_tab_released()

        elif self._state == _TabState.COOLDOWN:
            if self._countdown > 0:
                self._countdown -= 1
            else:
                self._state = _TabState.IDLE

    def _on_tab_released(self) -> None:
        if not self._got_valid:
            print_status("Tab released without valid Tab screen capture.")
        self._state = _TabState.COOLDOWN
        self._countdown = self._cooldown_wait


class SubtitleSystem:
    """OCR subtitle detection, runs every N ticks."""

    def __init__(self, on_match, on_detected=None, on_hero_switch=None,
                 transcript: bool = False, detection_delay_ticks: int = 0):
        self._state = SubtitleState()
        self._on_match = on_match
        self._on_detected = on_detected  # immediate callback on detection
        self._on_hero_switch = on_hero_switch  # callback(player, hero, sim_time)
        self._detection_delay = detection_delay_ticks
        self._pending_detection: tuple[str, int] | None = None  # (result, trigger_tick)

        if transcript:
            transcript_dir = Path("transcripts")
            transcript_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._state.transcript_file = open(
                transcript_dir / f"{ts}.txt", "a", encoding="utf-8"
            )
            _logger.info(f"Transcript logging to transcripts/{ts}.txt")

    def on_tick(self, ctx: TickContext) -> None:
        # Check if pending detection delay has elapsed
        if self._pending_detection:
            pending_result, trigger_tick = self._pending_detection
            if ctx.tick - trigger_tick >= self._detection_delay:
                self._pending_detection = None
                self._on_match(pending_result, ctx.sim_time)

        # Snapshot hero_map before processing to detect new switches
        prev_map = dict(self._state.hero_map) if self._on_hero_switch else {}

        result = process_subtitle_frame(ctx.frame_bgr, ctx.sim_time, self._state)

        # Emit hero switch events for new or changed entries
        if self._on_hero_switch:
            for player, hero in self._state.hero_map.items():
                if prev_map.get(player) != hero:
                    self._on_hero_switch(player, hero, ctx.sim_time)

        if result:
            if self._on_detected:
                self._on_detected(result, ctx.sim_time)
            if self._detection_delay > 0 and self._pending_detection is None:
                self._pending_detection = (result, ctx.tick)
            else:
                self._on_match(result, ctx.sim_time)

    @property
    def hero_map(self) -> dict[str, str]:
        return dict(self._state.hero_map)

    @property
    def hero_history(self) -> dict[str, list[tuple[float, str]]]:
        return {k: list(v) for k, v in self._state.hero_history.items()}

    def flush_pending(self, sim_time: float) -> None:
        """Fire any pending detection immediately (e.g., when replay ends)."""
        if self._pending_detection:
            result, _ = self._pending_detection
            self._pending_detection = None
            self._on_match(result, sim_time)

    def reset_match(self) -> None:
        self._state.hero_map.clear()
        self._state.hero_history.clear()
        self._pending_detection = None

    def close(self) -> None:
        if self._state.transcript_file:
            self._state.transcript_file.close()
            self._state.transcript_file = None


class OverwolfSystem:
    """Drains Overwolf event queue each tick, writes to recording, dispatches to app."""

    def __init__(self, queue: OverwolfEventQueue,
                 on_event: Callable[[Any], None] | None = None,
                 writer: OverwolfRecordingWriter | None = None) -> None:
        self._queue = queue
        self._on_event = on_event
        self._writer = writer

    def set_writer(self, writer: OverwolfRecordingWriter | None) -> None:
        self._writer = writer

    def clear_writer(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None

    def on_tick(self, ctx: TickContext) -> None:
        events = self._queue.drain()
        for event in events:
            if self._writer:
                self._writer.write(event, ctx.tick)
            if self._on_event:
                try:
                    self._on_event(event)
                except Exception as e:
                    _logger.warning(f"Overwolf event handler error: {e}")


class ReplayOverwolfSource:
    """Feeds recorded Overwolf events into a queue at the right frame/tick."""

    def __init__(self, events: Sequence[tuple[int, Any]],
                 queue: OverwolfEventQueue) -> None:
        self._events = events
        self._queue = queue
        self._idx = 0

    def advance_to(self, tick: int) -> None:
        """Push all events up to and including this tick into the queue."""
        while self._idx < len(self._events):
            frame, event = self._events[self._idx]
            if frame > tick:
                break
            self._queue.push(event)
            self._idx += 1


class ChatSystem:
    """OCR chat detection for player join/leave events, runs every N ticks."""

    def __init__(self, on_player_change=None):
        self._state = ChatState()
        self._on_player_change = on_player_change  # callback(player, event, sim_time)
        self._last_count = 0  # track new events

    def on_tick(self, ctx: TickContext) -> None:
        process_chat_frame(ctx.frame_bgr, ctx.sim_time, self._state)

        # Emit callbacks for new events
        if self._on_player_change:
            while self._last_count < len(self._state.player_changes):
                sim_time, player, event = self._state.player_changes[self._last_count]
                self._on_player_change(player, event, sim_time)
                self._last_count += 1

    @property
    def player_changes(self) -> list[tuple[float, str, str]]:
        return list(self._state.player_changes)

    def reset_match(self) -> None:
        self._state.player_changes.clear()
        self._last_count = 0


class TeamSideSystem:
    """Detects ATTACK/DEFEND label in the top-left of Escort/Hybrid modes.

    Scans the top 20%, left 30% of each frame for pink (attack) or light blue
    (defend) pixels. When >2% coverage is found, runs Tesseract OCR on the
    binarized mask with a restricted character set and fuzzy-matches to
    ATTACK or DEFEND.

    Stops scanning once a match is found (one detection per match).
    """

    # Target colors (RGB) and tolerance
    _ATTACK_RGB = np.array([199, 13, 78], dtype=float)
    _DEFEND_RGB = np.array([137, 233, 253], dtype=float)
    _TOLERANCE = 40.0
    _MIN_COVERAGE = 0.02  # 2% of ROI pixels

    _SIDE_MODES = {"Escort", "Hybrid"}

    def __init__(self, app, on_detected: Callable[[str], None] | None = None):
        self._app = app
        self._on_detected = on_detected
        self._detected = False
        self._active = False  # scanning in progress

    def start(self) -> None:
        """Start scanning. Called on MatchStart."""
        self._active = True
        self._detected = False

    def stop(self) -> None:
        """Stop scanning."""
        self._active = False

    def reset_match(self) -> None:
        self._detected = False
        self._active = False

    def on_tick(self, ctx: TickContext) -> None:
        if self._detected or not self._active:
            return

        # Stop if local hero is already resolved (label is gone)
        if self._app is not None:
            ms = self._app._match_state
            local = ms.local_player
            if local and local.current_hero:
                _logger.info("TeamSide: local hero resolved, stopping scan")
                self._active = False
                return
            # Stop if mode resolved to something without sides
            if ms.mode and ms.mode not in self._SIDE_MODES:
                self._active = False
                return

        frame = ctx.frame_bgr
        h, w = frame.shape[:2]
        roi = frame[:int(h * 0.2), :int(w * 0.3)]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(float)

        for target, label in [(self._ATTACK_RGB, "ATTACK"),
                              (self._DEFEND_RGB, "DEFEND")]:
            diff = np.sqrt(np.sum((rgb - target) ** 2, axis=2))
            mask = (diff < self._TOLERANCE).astype(np.uint8) * 255
            coverage = float(np.mean(mask > 0))
            if coverage >= self._MIN_COVERAGE:
                result = self._ocr_mask(mask, label)
                if result:
                    self._detected = True
                    _logger.info(f"Team side detected: {result}")
                    if self._on_detected:
                        self._on_detected(result)
                    return

    def _ocr_mask(self, mask: np.ndarray, hint: str) -> str | None:
        """OCR the binarized mask and fuzzy-match to ATTACK or DEFEND."""
        import os
        from pytesseract_api.api import get_image_data, get_tess_lib
        from pytesseract_api.capi_types import TessPageSegMode
        from overwatchlooker.heroes import edit_distance

        tess_lib_path = r"C:\Program Files\Tesseract-OCR\libtesseract-5.dll"
        tess_data = r"C:\Program Files\Tesseract-OCR\tessdata"
        lib = get_tess_lib(tess_lib_path)
        api = lib.TessBaseAPICreate()
        lib.TessBaseAPIInit3(api, tess_data.encode(), b"eng")

        try:
            # Restrict to characters in ATTACK and DEFEND
            lib.TessBaseAPISetVariable(api, b"tessedit_char_whitelist",
                                       b"ATACKDEFNatckdefn")
            lib.TessBaseAPISetPageSegMode(api, TessPageSegMode.PSM_SINGLE_LINE.value)

            data = get_image_data(mask.copy())

            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            os.dup2(devnull, 2)
            try:
                lib.TessBaseAPISetImage(api, *data)
                res: bytes = lib.TessBaseAPIGetUTF8Text(api)
                text = res.decode().strip().upper()
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
                os.close(devnull)
        finally:
            lib.TessBaseAPIEnd(api)
            lib.TessBaseAPIDelete(api)

        if not text:
            return None

        # Fuzzy match to ATTACK or DEFEND
        d_attack = edit_distance(text, "ATTACK")
        d_defend = edit_distance(text, "DEFEND")
        best = min(d_attack, d_defend)
        if best > 2:
            _logger.debug(f"Team side OCR '{text}' too far from ATTACK/DEFEND")
            return None
        return "ATTACK" if d_attack <= d_defend else "DEFEND"


# ---------------------------------------------------------------------------
# Tick loop
# ---------------------------------------------------------------------------

class TickLoop:
    """Unified frame loop with barrier-synchronized parallel systems."""

    def __init__(self, fps: int, frame_source, input_source):
        self.fps = fps
        self.frame_source = frame_source
        self.input_source = input_source
        self._systems: list[tuple[int, object]] = []  # (interval, system_fn)
        self._pre_tick_hooks: list[Callable[[int], None]] = []
        self._threads: list[threading.Thread] = []
        self.running = True
        self._current_tick = 0

    def register(self, on_tick, every_n_ticks: int = 1) -> None:
        self._systems.append((every_n_ticks, on_tick))

    def register_pre_tick(self, hook) -> None:
        """Register a hook called with (tick) before systems run each frame."""
        self._pre_tick_hooks.append(hook)

    def run(self) -> None:
        n = len(self._systems)
        if n == 0:
            return

        barrier = threading.Barrier(n + 1)
        ctx_holder: list[TickContext | None] = [None]
        tick_holder = [0]

        for interval, system_fn in self._systems:
            t = threading.Thread(
                target=self._system_loop,
                args=(system_fn, interval, barrier, ctx_holder, tick_holder),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        tick = 0
        try:
            while self.running:
                frame = self.frame_source.next_frame()
                if frame is None:
                    if isinstance(self.frame_source, ReplayFrameSource):
                        break  # replay exhausted
                    continue  # Live: no frame, don't advance tick

                sim_time = tick / self.fps
                self.input_source.advance_to(tick)

                # Pre-tick hooks (e.g. replay overwolf source feeding queue)
                for hook in self._pre_tick_hooks:
                    hook(tick)

                self._current_tick = tick
                ctx_holder[0] = TickContext(tick, sim_time, frame, self.input_source)
                tick_holder[0] = tick

                try:
                    barrier.wait(timeout=2.0)  # release system threads
                    barrier.wait(timeout=2.0)  # wait for systems to finish
                except threading.BrokenBarrierError:
                    if not self.running:
                        break
                    barrier.reset()

                tick += 1
        finally:
            self.running = False
            barrier.abort()

    def _system_loop(self, system_fn, interval, barrier, ctx_holder, tick_holder):
        while self.running:
            try:
                barrier.wait(timeout=2.0)  # wait for main thread to post context
            except threading.BrokenBarrierError:
                if not self.running:
                    return
                continue

            if not self.running:
                return

            tick = tick_holder[0]
            if tick % interval == 0:
                try:
                    system_fn(ctx_holder[0])
                except Exception as e:
                    _logger.warning(f"System tick error: {e}")

            try:
                barrier.wait(timeout=2.0)  # signal done
            except threading.BrokenBarrierError:
                if not self.running:
                    return

    def stop(self) -> None:
        self.running = False
        self.input_source.stop()
