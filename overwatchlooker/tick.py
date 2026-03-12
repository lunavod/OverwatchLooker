"""Tick-based frame loop for both live and replay modes."""

import datetime
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from overwatchlooker.display import print_status
from overwatchlooker.screenshot import (
    crop_hero_panel,
    has_hero_panel,
    is_ow2_tab_screen_bgr,
    ocr_hero_name,
    save_screenshot,
)
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

class LiveFrameSource:
    """Captures screen via dxcam, paces at real FPS."""

    def __init__(self, fps: int):
        import dxcam
        self._camera = dxcam.create(output_color="BGR")
        self._fps = fps
        self._interval = 1.0 / fps
        self._last_time = 0.0
        # Start continuous capture — dxcam buffers the latest frame
        self._camera.start(target_fps=fps)

    def next_frame(self) -> np.ndarray | None:
        # Pace at FPS
        now = time.perf_counter()
        sleep_time = self._interval - (now - self._last_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.perf_counter()

        # Only capture when Overwatch is foreground
        from overwatchlooker.hotkey import _get_foreground_exe
        if _get_foreground_exe() != "overwatch.exe":
            return None

        try:
            frame = self._camera.get_latest_frame()
        except Exception:
            return None
        return frame


class ReplayFrameSource:
    """Reads frames sequentially from a FrameReader, no sleeps."""

    def __init__(self, reader):
        self._reader = reader

    def next_frame(self) -> np.ndarray | None:
        return self._reader.read_next()


# ---------------------------------------------------------------------------
# Input sources
# ---------------------------------------------------------------------------

class LiveInputSource:
    """Wraps pynput, tracks key state per tick."""

    def __init__(self):
        from pynput import keyboard

        self._held: set[str] = set()
        self._just_pressed: set[str] = set()
        self._just_released: set[str] = set()
        self._pending_press: list[str] = []
        self._pending_release: list[str] = []
        self._lock = threading.Lock()

        def _key_to_str(key) -> str:
            if isinstance(key, keyboard.Key):
                return key.name
            if isinstance(key, keyboard.KeyCode):
                if key.char:
                    return key.char
                if key.vk:
                    return f"vk_{key.vk}"
            return str(key)

        def on_press(key):
            name = _key_to_str(key)
            with self._lock:
                self._pending_press.append(name)

        def on_release(key):
            name = _key_to_str(key)
            with self._lock:
                self._pending_release.append(name)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def advance_to(self, tick: int) -> None:
        with self._lock:
            presses = self._pending_press
            releases = self._pending_release
            self._pending_press = []
            self._pending_release = []

        self._just_pressed = set(presses)
        self._just_released = set(releases)
        self._held.update(presses)
        self._held -= set(releases)

    def is_key_held(self, key: str) -> bool:
        return key in self._held

    def key_just_pressed(self, key: str) -> bool:
        return key in self._just_pressed

    def key_just_released(self, key: str) -> bool:
        return key in self._just_released

    def stop(self) -> None:
        self._listener.stop()


class ReplayInputSource:
    """Replays events from events.jsonl."""

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

        self._initial_wait = max(1, int(fps * 1.0))
        self._retry_wait = max(1, int(fps * 0.5))
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
                saved_path = save_screenshot(png_bytes)
                self._app.store_valid_tab(png_bytes, ctx.sim_time, saved_path.name)
                if has_hero_panel(png_bytes):
                    crop = crop_hero_panel(png_bytes)
                    name = ocr_hero_name(crop)
                    if name:
                        self._app.store_hero_crop(name, crop)
                self._got_valid = True
                self._state = _TabState.DONE
            else:
                _logger.info(f"Tab screen rejected: {reason}")
                self._state = _TabState.RETRY_WAIT
                self._countdown = self._retry_wait

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

    def __init__(self, on_match, transcript: bool = False):
        self._state = SubtitleState()
        self._on_match = on_match

        if transcript:
            transcript_dir = Path("transcripts")
            transcript_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._state.transcript_file = open(
                transcript_dir / f"{ts}.txt", "a", encoding="utf-8"
            )
            _logger.info(f"Transcript logging to transcripts/{ts}.txt")

    def on_tick(self, ctx: TickContext) -> None:
        result = process_subtitle_frame(ctx.frame_bgr, ctx.sim_time, self._state)
        if result:
            self._on_match(result, ctx.sim_time)

    @property
    def hero_map(self) -> dict[str, str]:
        return dict(self._state.hero_map)

    @property
    def hero_history(self) -> dict[str, list[tuple[float, str]]]:
        return {k: list(v) for k, v in self._state.hero_history.items()}

    def reset_match(self) -> None:
        self._state.hero_map.clear()
        self._state.hero_history.clear()

    def close(self) -> None:
        if self._state.transcript_file:
            self._state.transcript_file.close()
            self._state.transcript_file = None


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
        self._threads: list[threading.Thread] = []
        self.running = True
        self._current_tick = 0
        self.on_frame = None  # optional callback(frame_bgr) called for every captured frame
        self.on_key_events = None  # optional callback(tick, pressed, released) for recording

    def register(self, on_tick, every_n_ticks: int = 1) -> None:
        self._systems.append((every_n_ticks, on_tick))

    def run(self) -> None:
        n = len(self._systems)
        if n == 0:
            return

        barrier = threading.Barrier(n + 1)
        ctx_holder = [None]
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

                if self.on_key_events:
                    pressed = self.input_source._just_pressed
                    released = self.input_source._just_released
                    if pressed or released:
                        self.on_key_events(tick, pressed, released)

                if self.on_frame:
                    self.on_frame(frame)
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
        if isinstance(self.frame_source, LiveFrameSource):
            try:
                self.frame_source._camera.stop()
            except Exception:
                pass
