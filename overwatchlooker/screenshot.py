import ctypes
import ctypes.wintypes
import io
from datetime import datetime
from pathlib import Path

import mss
import mss.tools
from PIL import Image

from overwatchlooker.config import MONITOR_INDEX

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


def capture_monitor() -> bytes:
    """Capture the primary monitor and return PNG bytes."""
    with mss.mss() as sct:
        monitor = sct.monitors[MONITOR_INDEX]
        img = sct.grab(monitor)
        png_bytes = mss.tools.to_png(img.rgb, img.size)
        return _resize_if_needed(png_bytes)
