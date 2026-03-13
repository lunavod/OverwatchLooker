"""Visual subtitle-based VICTORY/DEFEAT detection via screen capture + OCR."""

import datetime
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import os
import sys

import cv2
import numpy as np
from pytesseract_api import image_to_string as _tess_image_to_string, TessPageSegMode

from overwatchlooker.hotkey import _get_foreground_exe
from overwatchlooker.config import (
    AUDIO_COOLDOWN_SECONDS,
    MONITOR_INDEX,
    SUBTITLE_POLL_INTERVAL,
)

_logger = logging.getLogger("overwatchlooker")

# Tesseract C API paths (ctypes, no subprocess)
_TESS_LIB = r"C:\Program Files\Tesseract-OCR\libtesseract-5.dll"
_TESS_DATA = r"C:\Program Files\Tesseract-OCR\tessdata"

# Open /dev/null once for stderr suppression
_devnull_fd = os.open(os.devnull, os.O_WRONLY)


def _ocr(binary: np.ndarray) -> str:
    """Run Tesseract on a binary image via C API. Returns lowercase text."""
    # Suppress Tesseract's noisy C-level stderr warnings
    old_stderr = os.dup(2)
    os.dup2(_devnull_fd, 2)
    try:
        return _tess_image_to_string(
            binary, psm=TessPageSegMode.PSM_SINGLE_BLOCK,
            lib_path=_TESS_LIB, tessdata_path=_TESS_DATA,
        ).strip().lower()
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

# Subtitle region: bottom of screen, above HUD ability bar
_REGION_Y_START = 0.88
_REGION_Y_END = 0.98
_REGION_X_START = 0.30
_REGION_X_END = 0.70

# HSV thresholds for bright/saturated text pixels
_VALUE_MIN = 180          # minimum brightness for any text color
_SAT_MAX_WHITE = 30       # max saturation for white text
_SAT_MIN_COLOR = 80       # min saturation for colored text (red/blue/green)
_PIXEL_THRESHOLD = 500    # minimum text pixels to trigger OCR


from overwatchlooker.heroes import edit_distance, match_hero_name

# Keep _edit_distance as alias for backward compat (imported by tray.py, tests)
_edit_distance = edit_distance


class SubtitleState:
    """Mutable state for subtitle processing."""

    def __init__(self):
        self.last_trigger_time: float = float("-inf")
        self.hero_map: dict[str, str] = {}  # UPPERCASE username -> hero name
        self.hero_history: dict[str, list[tuple[float, str]]] = {}  # username -> [(sim_time, hero)]
        self.last_lines: set[str] = set()  # transcript dedup
        self.transcript_file = None  # IO file or None


def process_subtitle_frame(frame_bgr: np.ndarray, sim_time: float,
                           state: SubtitleState) -> str | None:
    """Process full frame for subtitles. Returns 'VICTORY'/'DEFEAT' or None.

    Mutates state. Extracts subtitle region internally.
    """
    h, w = frame_bgr.shape[:2]
    img = frame_bgr[int(h * _REGION_Y_START):int(h * _REGION_Y_END),
                     int(w * _REGION_X_START):int(w * _REGION_X_END)]

    # Cooldown check
    if sim_time - state.last_trigger_time < AUDIO_COOLDOWN_SECONDS:
        return None

    # Stage 1: fast pixel check
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2] >= _VALUE_MIN
    white_mask = bright & (hsv[:, :, 1] <= _SAT_MAX_WHITE)
    color_mask = bright & (hsv[:, :, 1] >= _SAT_MIN_COLOR)
    text_mask = white_mask | color_mask
    text_count = int(np.count_nonzero(text_mask))

    if text_count < _PIXEL_THRESHOLD:
        return None

    _logger.debug(f"Subtitle: {text_count} text pixels detected, running OCR...")

    # Stage 2: binarize and OCR
    binary = np.zeros(img.shape[:2], dtype=np.uint8)
    binary[text_mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)

    text = _ocr(binary)
    _logger.debug(f"Subtitle OCR text: {text!r}")

    # Extract username -> hero mappings
    frame_heroes: dict[str, str] = {}
    for m in re.finditer(r"\[(\w+)\s+\(([^)]+)\)\]", text):
        username = m.group(1).upper()
        raw_hero = m.group(2).strip().title()
        if username != "ATHENA":
            hero = match_hero_name(raw_hero) or raw_hero
            frame_heroes[username] = hero

    for username, hero in frame_heroes.items():
        history = state.hero_history.get(username)
        if history:
            last_hero = history[-1][1]
            if _edit_distance(hero.lower(), last_hero.lower()) <= 2:
                continue
        if username not in state.hero_history:
            state.hero_history[username] = []
        state.hero_history[username].append((sim_time, hero))
        state.hero_map[username] = hero

    # Transcript dedup
    if state.transcript_file and text:
        current_lines = set()
        for ln in text.splitlines():
            ln = ln.strip()
            if ln and re.match(r"\[.+\]", ln):
                clean = re.sub(r"\s+[^\w\s()\[\]!?.,']{1,3}(\s+[^\w\s()\[\]!?.,']{1,3})*\s*$", "", ln)
                current_lines.add(clean)
        new_lines = current_lines - state.last_lines
        state.last_lines = current_lines
        if new_lines:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            for ln in sorted(new_lines):
                state.transcript_file.write(f"[{ts}] {ln}\n")
            state.transcript_file.flush()

    # Match Athena lines
    result = None
    if re.search(r"athena\W{0,3}\s*victory", text):
        result = "VICTORY"
    elif re.search(r"athena\W{0,3}\s*defeat", text):
        result = "DEFEAT"

    if result:
        _logger.info(f"Subtitle detected: {result}")
        state.last_trigger_time = sim_time

    return result


class SubtitleListener:
    """Monitors screen for VICTORY/DEFEAT subtitle text."""

    def __init__(self, on_match: Callable[[str], None], transcript: bool = False,
                 screen_provider=None, clock: Callable[[], float] | None = None):
        self._on_match = on_match
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_trigger_time = 0.0
        self._transcript = transcript
        self._transcript_file = None
        self._last_lines: set[str] = set()  # lines from the previous OCR pass
        self._hero_map: dict[str, str] = {}  # UPPERCASE username -> hero name (Title Case)
        self._hero_history: dict[str, list[tuple[float, str]]] = {}  # username -> [(time, hero), ...]
        self._screen_provider = screen_provider  # optional ReplaySource for replay mode
        self._clock = clock or time.monotonic

    @property
    def hero_map(self) -> dict[str, str]:
        """Username -> hero name mapping from subtitle OCR. Usernames are UPPERCASE."""
        return dict(self._hero_map)

    @property
    def hero_history(self) -> dict[str, list[tuple[float, str]]]:
        """Username -> list of (monotonic_time, hero_name) tracking hero switches."""
        return {k: list(v) for k, v in self._hero_history.items()}

    def reset_match(self) -> None:
        """Clear hero tracking for a new match."""
        self._hero_map.clear()
        self._hero_history.clear()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        if self._transcript:
            transcript_dir = Path("transcripts")
            transcript_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._transcript_file = open(transcript_dir / f"{ts}.txt", "a", encoding="utf-8")
            _logger.info(f"Transcript logging to transcripts/{ts}.txt")

        _logger.info("Subtitle listener started.")

        try:
            if self._screen_provider:
                # Replay mode: use replay-aware sleep so polls scale with speed
                while self._running:
                    try:
                        self._poll(None)
                    except Exception as e:
                        _logger.warning(f"Subtitle poll error: {e}")
                    self._screen_provider.sleep(SUBTITLE_POLL_INTERVAL)
            else:
                import mss
                with mss.mss() as sct:
                    while self._running:
                        try:
                            self._poll(sct)
                        except Exception as e:
                            _logger.warning(f"Subtitle poll error: {e}")
                        time.sleep(SUBTITLE_POLL_INTERVAL)
        finally:
            if self._transcript_file:
                self._transcript_file.close()
                self._transcript_file = None

        _logger.info("Subtitle listener stopped.")

    def _poll(self, sct) -> None:
        # Only poll when Overwatch is the active window (skip check in replay mode)
        if not self._screen_provider and _get_foreground_exe() != "overwatch.exe":
            return

        # Cooldown check
        now = self._clock()
        if now - self._last_trigger_time < AUDIO_COOLDOWN_SECONDS:
            return

        # Capture subtitle region
        if self._screen_provider:
            img = self._screen_provider.get_region_bgr(
                _REGION_X_START, _REGION_Y_START, _REGION_X_END, _REGION_Y_END
            )
        else:
            monitor = sct.monitors[MONITOR_INDEX]
            x1 = int(monitor["left"] + monitor["width"] * _REGION_X_START)
            x2 = int(monitor["left"] + monitor["width"] * _REGION_X_END)
            y1 = int(monitor["top"] + monitor["height"] * _REGION_Y_START)
            y2 = int(monitor["top"] + monitor["height"] * _REGION_Y_END)

            region = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
            grab = sct.grab(region)
            img = np.array(grab)[:, :, :3]  # drop alpha, keep BGR

        # Stage 1: fast pixel check — detect bright white OR saturated color text
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bright = hsv[:, :, 2] >= _VALUE_MIN
        white_mask = bright & (hsv[:, :, 1] <= _SAT_MAX_WHITE)
        color_mask = bright & (hsv[:, :, 1] >= _SAT_MIN_COLOR)
        text_mask = white_mask | color_mask
        text_count = int(np.count_nonzero(text_mask))

        if text_count < _PIXEL_THRESHOLD:
            return

        _logger.debug(f"Subtitle: {text_count} text pixels detected, running OCR...")

        # Stage 2: binarize using the text mask and run Tesseract
        binary = np.zeros(img.shape[:2], dtype=np.uint8)
        binary[text_mask] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.dilate(binary, kernel, iterations=1)

        text = _ocr(binary)
        _logger.debug(f"Subtitle OCR text: {text!r}")

        # Extract username -> hero mappings from [USERNAME (HERO)] lines
        # Take only the last hero per player in this frame (newest subtitle)
        now = self._clock()
        frame_heroes: dict[str, str] = {}
        for m in re.finditer(r"\[(\w+)\s+\(([^)]+)\)\]", text):
            username = m.group(1).upper()
            raw_hero = m.group(2).strip().title()
            if username != "ATHENA":
                # Fuzzy-match to canonical hero name
                hero = match_hero_name(raw_hero) or raw_hero
                frame_heroes[username] = hero

        for username, hero in frame_heroes.items():
            history = self._hero_history.get(username)
            if history:
                last_hero = history[-1][1]
                if _edit_distance(hero.lower(), last_hero.lower()) <= 2:
                    continue  # OCR artifact, same hero
            # New player or genuine hero switch
            if username not in self._hero_history:
                self._hero_history[username] = []
            self._hero_history[username].append((now, hero))
            self._hero_map[username] = hero

        # Write new subtitle lines to transcript (deduplicated against previous pass)
        if self._transcript_file and text:
            # Only keep lines that look like subtitles: [SPEAKER] text
            current_lines = set()
            for ln in text.splitlines():
                ln = ln.strip()
                if ln and re.match(r"\[.+\]", ln):
                    # Strip trailing noise (random chars after the real text)
                    clean = re.sub(r"\s+[^\w\s()\[\]!?.,']{1,3}(\s+[^\w\s()\[\]!?.,']{1,3})*\s*$", "", ln)
                    current_lines.add(clean)
            new_lines = current_lines - self._last_lines
            self._last_lines = current_lines
            if new_lines:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                for ln in sorted(new_lines):
                    self._transcript_file.write(f"[{ts}] {ln}\n")
                self._transcript_file.flush()

        # Match Athena subtitle lines.
        # The game shows "[ATHENA] Victory." and "[ATHENA] Defeat."
        # OCR may drop brackets/punctuation, so we fuzzy-match the pattern.
        result = None
        if re.search(r"athena\W{0,3}\s*victory", text):
            result = "VICTORY"
        elif re.search(r"athena\W{0,3}\s*defeat", text):
            result = "DEFEAT"

        if result:
            _logger.info(f"Subtitle detected: {result}")
            self._last_trigger_time = self._clock()
            self._on_match(result)
