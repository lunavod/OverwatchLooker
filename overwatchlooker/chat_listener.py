"""Chat OCR for detecting player join/leave events in the in-game text chat."""

import logging
import re

import cv2
import numpy as np

from overwatchlooker.heroes import edit_distance
from overwatchlooker.subtitle_listener import _ocr

_logger = logging.getLogger("overwatchlooker")

# Chat region: bottom-left of screen
_REGION_Y_START = 0.46
_REGION_Y_END = 0.66
_REGION_X_START = 0.0
_REGION_X_END = 0.30

# HSV thresholds for yellow chat text
_HUE_MIN = 15
_HUE_MAX = 35
_SAT_MIN = 80
_VALUE_MIN = 150

_PIXEL_THRESHOLD = 300  # minimum yellow text pixels to trigger OCR

# Regex for join/leave messages: [username] joined/left the game
_JOIN_LEAVE_RE = re.compile(r"\[(\w+)\]\s*(?:joined|left)\s+the\s+game", re.IGNORECASE)


class ChatState:
    """Mutable state for chat processing."""

    def __init__(self):
        # [(sim_time, player_name, "joined"/"left")]
        self.player_changes: list[tuple[float, str, str]] = []
        self.last_lines: set[str] = set()  # dedup against previous frame


def _is_duplicate(username: str, event: str,
                  existing: list[tuple[float, str, str]]) -> bool:
    """Check if this event is a fuzzy duplicate of an already-recorded event."""
    for _, prev_name, prev_event in existing:
        if prev_event == event and edit_distance(username.lower(), prev_name.lower()) <= 2:
            return True
    return False


def process_chat_frame(frame_bgr: np.ndarray, sim_time: float,
                       state: ChatState) -> None:
    """Process full frame for chat join/leave events. Mutates state."""
    h, w = frame_bgr.shape[:2]
    img = frame_bgr[int(h * _REGION_Y_START):int(h * _REGION_Y_END),
                     int(w * _REGION_X_START):int(w * _REGION_X_END)]

    # Stage 1: fast pixel check — detect yellow chat text
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = (
        (hsv[:, :, 0] >= _HUE_MIN) & (hsv[:, :, 0] <= _HUE_MAX)
        & (hsv[:, :, 1] >= _SAT_MIN)
        & (hsv[:, :, 2] >= _VALUE_MIN)
    )
    text_count = int(np.count_nonzero(yellow_mask))

    if text_count < _PIXEL_THRESHOLD:
        return

    _logger.debug(f"Chat: {text_count} yellow text pixels detected, running OCR...")

    # Stage 2: binarize and OCR
    binary = np.zeros(img.shape[:2], dtype=np.uint8)
    binary[yellow_mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)  # type: ignore[assignment]

    text = _ocr(binary)
    _logger.debug(f"Chat OCR text: {text!r}")

    # Extract join/leave events
    for m in _JOIN_LEAVE_RE.finditer(text):
        username = m.group(1).upper()
        event = "joined" if "joined" in m.group(0).lower() else "left"

        # Fuzzy dedup: skip if a similar event already exists
        if _is_duplicate(username, event, state.player_changes):
            continue

        state.player_changes.append((sim_time, username, event))
        _logger.info(f"Chat: {username} {event} the game at {sim_time:.1f}s")
