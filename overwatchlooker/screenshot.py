import io

import mss
import mss.tools
from PIL import Image

from overwatchlooker.config import MONITOR_INDEX

MAX_PNG_SIZE = 4_000_000  # 4MB, stay under Claude's 5MB limit
MAX_LONG_EDGE = 1568  # Claude's optimal image size


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
