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


class ControlScoreSystem:
    """Detects Control/Flashpoint round score from the top-center HUD.

    Looks for filled blue/red score indicators (circles for Control,
    diamonds for Flashpoint). Runs every 2 seconds.
    Results within 5s before or 15s after a death event are invalidated
    (kill cam shows misleading HUD).
    """

    # Fixed indicator center positions (fractions of screen size)
    # Flashpoint: 3 diamonds per side
    _FP_BLUE_X = [0.3594, 0.3812, 0.4031]
    _FP_RED_X = [0.5969, 0.6188, 0.6406]
    _FP_Y = [0.1069, 0.0699]  # with banner (lower) / without banner (higher)
    # Control: 2 circles per side
    _CTRL_BLUE_X = [0.384, 0.413]
    _CTRL_RED_X = [0.587, 0.616]
    _CTRL_Y = [0.0769]  # TODO: find banner/no-banner Y positions

    # Patch sampling: (2*R+1)x(2*R+1) patch at each position
    _PATCH_R = 2
    _MAX_STD = 5.0  # max temporal std — HUD center is opaque (std~0), background shifts

    # Frame buffer for temporal variance check
    _BUFFER_SIZE = 10

    # Death invalidation windows (seconds)
    _DEATH_PRE = 5.0
    _DEATH_POST = 15.0

    def __init__(self, app: Any, on_score_change: Callable[[list[tuple[int, int]]], None] | None = None,
                 debug: bool = False):
        self._app = app
        self._on_score_change = on_score_change
        self._debug = debug
        self._active = False
        self._max_score = 2
        self._blue_x: list[float] = []
        self._red_x: list[float] = []
        self._death_times: list[float] = []
        self._last_confirmed: tuple[int, int] = (0, 0)
        self._transitions: list[tuple[int, int]] = [(0, 0)]
        self._frame_buffer: list[np.ndarray] = []

    def start(self, max_score: int = 2) -> None:
        self._active = True
        self._max_score = max_score
        if max_score == 3:  # Flashpoint
            self._blue_x = self._FP_BLUE_X
            self._red_x = self._FP_RED_X
            self._y_positions = self._FP_Y
        else:  # Control
            self._blue_x = self._CTRL_BLUE_X
            self._red_x = self._CTRL_RED_X
            self._y_positions = self._CTRL_Y
        self._frame_buffer.clear()

    def stop(self) -> None:
        self._active = False

    def reset_match(self) -> None:
        self._active = False
        self._max_score = 2
        self._blue_x = []
        self._red_x = []
        self._y_positions = []
        self._death_times.clear()
        self._last_confirmed = (0, 0)
        self._transitions = [(0, 0)]
        self._frame_buffer.clear()

    def record_death(self, sim_time: float) -> None:
        """Record a death event time for invalidation."""
        self._death_times.append(sim_time)

    def _is_valid_time(self, t: float) -> bool:
        """Check if a sim_time is outside all death invalidation windows."""
        for dt in self._death_times:
            if dt - self._DEATH_PRE <= t <= dt + self._DEATH_POST:
                return False
        return True

    def _is_filled(self, fx: float, color: str, debug: bool = False) -> bool:
        """Check if an indicator at screen fraction (fx, fy) is filled.

        Samples a small patch at each Y position across all buffered frames.
        A filled HUD indicator has:
        - Correct HSV color (blue or red)
        - Near-zero temporal variance (opaque HUD center is identical every frame)
        """
        h, w = self._frame_buffer[0].shape[:2]
        r = self._PATCH_R
        cx = int(fx * w)

        for fy in self._y_positions:
            cy = int(fy * h)
            if cy - r < 0 or cy + r + 1 > h or cx - r < 0 or cx + r + 1 > w:
                continue

            # Temporal variance check
            patches = [f[cy - r:cy + r + 1, cx - r:cx + r + 1]
                       for f in self._frame_buffer]
            stack = np.stack(patches, axis=0).astype(float)
            std = float(np.mean(np.std(stack, axis=0)))
            if std > self._MAX_STD:
                if debug:
                    _logger.info(f"  {color} @ ({fx:.3f},{fy:.4f}) px=({cx},{cy}): "
                                 f"std={std:.1f} > {self._MAX_STD} SKIP")
                continue

            # Color check on middle frame
            mid_hsv = cv2.cvtColor(
                self._frame_buffer[len(self._frame_buffer) // 2],
                cv2.COLOR_BGR2HSV)
            patch = mid_hsv[cy - r:cy + r + 1, cx - r:cx + r + 1]
            s = float(np.mean(patch[:, :, 1]))
            v = float(np.mean(patch[:, :, 2]))
            hue = float(np.mean(patch[:, :, 0]))

            if debug:
                _logger.info(f"  {color} @ ({fx:.3f},{fy:.4f}) px=({cx},{cy}): "
                             f"std={std:.1f} hue={hue:.0f} s={s:.0f} v={v:.0f}")

            if color == "blue" and s > 170 and v > 150 and 80 < hue < 120:
                return True
            if color == "red" and s > 150 and v > 150 and (hue > 140 or hue < 15):
                return True

        return False

    def on_tick(self, ctx: TickContext) -> None:
        if not self._active:
            return

        # Don't scan until at least one round has started
        if self._app is not None:
            ms = self._app._match_state
            if not ms.rounds:
                return

        # Buffer full frames (we sample specific positions, not ROIs)
        self._frame_buffer.append(ctx.frame_bgr.copy())
        if len(self._frame_buffer) > self._BUFFER_SIZE:
            self._frame_buffer.pop(0)
        if len(self._frame_buffer) < self._BUFFER_SIZE:
            return

        debug = self._debug
        if debug:
            _logger.info(f"ControlScore tick={ctx.tick} t={ctx.sim_time:.1f}s:")
        blue_score = min(
            sum(1 for fx in self._blue_x if self._is_filled(fx, "blue", debug)),
            self._max_score)
        red_score = min(
            sum(1 for fx in self._red_x if self._is_filled(fx, "red", debug)),
            self._max_score)

        # Only accept if this reading is valid (not in death window)
        if not self._is_valid_time(ctx.sim_time):
            if debug:
                _logger.info(f"  => {blue_score}-{red_score} INVALID (death window)")
            return

        score = (blue_score, red_score)
        if debug:
            _logger.info(f"  => {blue_score}-{red_score} (last={self._last_confirmed})")
        if score != self._last_confirmed:
            # Score can only go up, never down
            if (score[0] >= self._last_confirmed[0]
                    and score[1] >= self._last_confirmed[1]):
                self._last_confirmed = score
                self._transitions.append(score)
                _logger.info(f"Score: {score[0]}-{score[1]} "
                             f"(tick={ctx.tick}, t={ctx.sim_time:.1f}s)")
                if self._on_score_change:
                    self._on_score_change(list(self._transitions))

    @property
    def transitions(self) -> list[tuple[int, int]]:
        return list(self._transitions)

    def format_transitions(self) -> str:
        return " -> ".join(f"{b}:{r}" for b, r in self._transitions)


class TeamSideSystem:
    """Detects ATTACK/DEFEND label in the top-left of Escort/Hybrid modes.

    Scans the top 20%, left 30% of each frame for pink (attack) or light blue
    (defend) pixels. When >2% coverage is found, runs Tesseract OCR on the
    binarized mask with a restricted character set and fuzzy-matches to
    ATTACK or DEFEND using a finetuned PaddleOCR model.

    Requires 3 consecutive agreeing reads to confirm (filters hallucinations
    on blank/sky frames).

    Stops scanning once confirmed or when a round starts.
    """

    _SIDE_MODES = {"Escort", "Hybrid"}
    _REQUIRED_AGREES = 3  # consecutive matching reads to confirm
    _MIN_CONFIDENCE = 0.90  # baseline confidence filter
    _START_DELAY = 3.0  # skip first N seconds (loading transition hallucinations)

    def __init__(self, app, on_detected: Callable[[str], None] | None = None):
        self._app = app
        self._on_detected = on_detected
        self._detected = False
        self._active = False
        self._started_at: float = 0.0
        self._consecutive: list[str] = []

    def start(self) -> None:
        """Start scanning. Called on MatchStart."""
        self._active = True
        self._detected = False
        self._started_at = 0.0  # set on first tick
        self._consecutive.clear()

    def stop(self) -> None:
        """Stop scanning."""
        self._active = False

    def reset_match(self) -> None:
        self._detected = False
        self._active = False
        self._started_at = 0.0
        self._consecutive.clear()

    def on_tick(self, ctx: TickContext) -> None:
        if self._detected or not self._active:
            return

        # Record start time and skip first few seconds
        if self._started_at == 0.0:
            self._started_at = ctx.sim_time
        if ctx.sim_time - self._started_at < self._START_DELAY:
            return

        # Stop conditions
        if self._app is not None:
            ms = self._app._match_state
            if ms.rounds:
                _logger.info("TeamSide: round started, stopping scan")
                self._active = False
                return
            if ms.mode and ms.mode not in self._SIDE_MODES:
                self._active = False
                return

        from overwatchlooker.hero_panel import detect_team_side

        result = detect_team_side(ctx.frame_bgr)
        if result is None:
            self._consecutive.clear()
            return

        side, conf = result
        _logger.debug(f"TeamSide read: {side} ({conf:.3f}) "
                      f"streak={len(self._consecutive)} tick={ctx.tick}")

        if conf < self._MIN_CONFIDENCE:
            self._consecutive.clear()
            return

        # Track consecutive agreement
        if self._consecutive and self._consecutive[-1] != side:
            self._consecutive.clear()
        self._consecutive.append(side)

        if len(self._consecutive) >= self._REQUIRED_AGREES:
            self._detected = True
            _logger.info(f"Team side detected: {side} "
                         f"({self._REQUIRED_AGREES} consecutive reads)")
            if self._on_detected:
                self._on_detected(side)


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
        self.max_ticks: int | None = None  # stop after this many ticks
        self.start_tick: int = 0  # skip ticks before this (replay seek)

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

        # Seek past skipped frames for replay start offset
        if self.start_tick > 0 and isinstance(self.frame_source, ReplayFrameSource):
            self.frame_source._reader.seek(self.start_tick)
            tick = self.start_tick
            self.input_source.advance_to(tick)
            # Flush pre-tick hooks up to the start tick. This is critical:
            # ReplayOverwolfSource.advance_to(tick) pushes ALL recorded events
            # with frame <= tick into the queue. On the first real tick,
            # OverwolfSystem drains and processes them all — so match-level
            # state (game_type, map, mode, roster, etc.) is fully populated
            # even when replaying from the middle of a recording.
            # Without this, systems that check e.g. game_type == RANKED at
            # match_ended would see None because those events fired at frame 0.
            for hook in self._pre_tick_hooks:
                hook(tick)
            _logger.info(f"Replay: seeked to tick {tick} "
                         f"({tick / self.fps:.1f}s)")

        try:
            while self.running:
                frame = self.frame_source.next_frame()
                if frame is None:
                    if isinstance(self.frame_source, ReplayFrameSource):
                        break  # replay exhausted
                    continue  # Live: no frame, don't advance tick

                if self.max_ticks is not None and tick >= self.max_ticks:
                    break

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
