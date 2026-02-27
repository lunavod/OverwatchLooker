"""Visual subtitle-based VICTORY/DEFEAT detection via screen capture + OCR."""

import logging
import re
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from overwatchlooker.hotkey import _get_foreground_exe
from overwatchlooker.config import (
    AUDIO_COOLDOWN_SECONDS,
    MONITOR_INDEX,
    SUBTITLE_POLL_INTERVAL,
)

_logger = logging.getLogger("overwatchlooker")

# Subtitle region: bottom 10% of screen, center 20%
_REGION_Y_START = 0.90
_REGION_Y_END = 1.00
_REGION_X_START = 0.40
_REGION_X_END = 0.60

# HSV thresholds for white text pixels (high value, low saturation)
_WHITE_V_MIN = 200
_WHITE_S_MAX = 30
_WHITE_PIXEL_THRESHOLD = 500  # minimum white pixels to trigger OCR

# Lazy-loaded EasyOCR reader singleton
_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                _logger.info("Loading EasyOCR reader for subtitle detection...")
                _reader = easyocr.Reader(["en"], gpu=True)
                _logger.info("EasyOCR reader loaded.")
    return _reader


class SubtitleListener:
    """Monitors screen for VICTORY/DEFEAT subtitle text."""

    def __init__(self, on_match: Callable[[str], None]):
        self._on_match = on_match
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_trigger_time = 0.0

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
        import mss

        _logger.info("Subtitle listener started.")

        with mss.mss() as sct:
            while self._running:
                try:
                    self._poll(sct)
                except Exception as e:
                    _logger.warning(f"Subtitle poll error: {e}")
                time.sleep(SUBTITLE_POLL_INTERVAL)

        _logger.info("Subtitle listener stopped.")

    def _poll(self, sct) -> None:
        # Only poll when Overwatch is the active window
        if _get_foreground_exe() != "overwatch.exe":
            return

        # Cooldown check
        now = time.monotonic()
        if now - self._last_trigger_time < AUDIO_COOLDOWN_SECONDS:
            return

        # Capture subtitle region
        monitor = sct.monitors[MONITOR_INDEX]
        x1 = int(monitor["left"] + monitor["width"] * _REGION_X_START)
        x2 = int(monitor["left"] + monitor["width"] * _REGION_X_END)
        y1 = int(monitor["top"] + monitor["height"] * _REGION_Y_START)
        y2 = int(monitor["top"] + monitor["height"] * _REGION_Y_END)

        region = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
        grab = sct.grab(region)
        img = np.array(grab)[:, :, :3]  # drop alpha, keep BGR

        # Stage 1: fast white pixel check via HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask = (hsv[:, :, 2] >= _WHITE_V_MIN) & (hsv[:, :, 1] <= _WHITE_S_MAX)
        white_count = int(np.count_nonzero(white_mask))

        if white_count < _WHITE_PIXEL_THRESHOLD:
            return

        _logger.info(f"Subtitle: {white_count} white pixels detected, running OCR...")

        # Stage 2: OCR confirmation
        reader = _get_reader()
        results = reader.readtext(img, detail=0)
        text = " ".join(results).lower()
        _logger.info(f"Subtitle OCR text: {text!r}")

        # Match exact Athena subtitle lines only.
        # The game shows "[ATHENA] Victory." and "[ATHENA] Defeat."
        # OCR may drop brackets/punctuation, so we fuzzy-match the pattern.
        result = None
        if re.search(r"athena\W{0,3}\s*victory", text):
            result = "VICTORY"
        elif re.search(r"athena\W{0,3}\s*defeat", text):
            result = "DEFEAT"

        if result:
            _logger.info(f"Subtitle detected: {result}")
            self._last_trigger_time = time.monotonic()
            self._on_match(result)
