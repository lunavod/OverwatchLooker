import ctypes
import ctypes.wintypes
import io
from datetime import datetime
from pathlib import Path

import cv2
import mss
import mss.tools
import numpy as np
from PIL import Image

from overwatchlooker.config import MONITOR_INDEX, SCREENSHOT_MAX_AGE_SECONDS

MAX_PNG_SIZE = 4_000_000  # 4MB limit for API payload
MAX_LONG_EDGE = 1568  # optimal image size for vision models
CSIDL_MYPICTURES = 0x0027


def _get_pictures_folder() -> Path:
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_MYPICTURES, None, 0, buf)
    return Path(buf.value)


def get_screenshots_dir() -> Path:
    folder = _get_pictures_folder() / "OverwatchLooker"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_screenshot(png_bytes: bytes) -> Path:
    """Save screenshot to Pictures/OverwatchLooker with a timestamp filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = get_screenshots_dir() / f"{timestamp}.png"
    path.write_bytes(png_bytes)
    return path


def _resize_if_needed(png_bytes: bytes) -> bytes:
    if len(png_bytes) <= MAX_PNG_SIZE:
        return png_bytes
    img = Image.open(io.BytesIO(png_bytes))
    img.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_latest_screenshot() -> Path | None:
    """Return the most recent screenshot, or None if stale/missing."""
    screenshots_dir = get_screenshots_dir()
    pngs = sorted(screenshots_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pngs:
        return None
    latest = pngs[0]
    age = (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).total_seconds()
    if age > SCREENSHOT_MAX_AGE_SECONDS:
        return None
    return latest


def is_ow2_tab_screen(png_bytes: bytes) -> bool:
    """Fast check whether a screenshot is an OW2 Tab scoreboard.

    The Tab screen always has a solid-colored panel across the top.
    We check that the left 30% of the top panel (y 2-6%) is a single
    uniform color (≤2 unique colors to allow minor compression artifacts).
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]

    # Middle 30% of top panel (y 2-6%) — avoids tabs on left and map text on right
    strip = img[int(0.02 * h):int(0.06 * h), int(0.35 * w):int(0.65 * w)]
    unique = len(np.unique(strip.reshape(-1, 3), axis=0))
    return unique <= 2


def capture_monitor() -> bytes:
    """Capture the primary monitor and return PNG bytes."""
    with mss.mss() as sct:
        monitor = sct.monitors[MONITOR_INDEX]
        img = sct.grab(monitor)
        png_bytes = mss.tools.to_png(img.rgb, img.size)
        return _resize_if_needed(png_bytes)
