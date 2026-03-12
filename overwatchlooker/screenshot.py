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

_user32 = ctypes.windll.user32

# Make process DPI-aware so window rects are in physical pixels (matching mss)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        _user32.SetProcessDPIAware()
    except Exception:
        pass

CSIDL_MYPICTURES = 0x0027

# Per-backend optimal long-edge sizes (images larger than this get downscaled
# server-side anyway, so sending bigger just wastes bandwidth/tokens).
ANALYZER_MAX_EDGE = {
    "anthropic": 1568,
    "codex": None,  # subscription-based, send full res for best OCR accuracy
    "ocr": None,  # no resize — OCR works on raw pixels
}


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


def resize_for_analyzer(png_bytes: bytes, analyzer: str) -> bytes:
    """Downscale PNG bytes to the optimal size for the given analyzer backend.

    Returns the original bytes unchanged if no resize is needed.
    """
    max_edge = ANALYZER_MAX_EDGE.get(analyzer)
    if max_edge is None:
        return png_bytes
    img = Image.open(io.BytesIO(png_bytes))
    if max(img.size) <= max_edge:
        return png_bytes
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
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


def is_ow2_tab_screen(png_bytes: bytes) -> bool | str:
    """Fast check whether a screenshot is an OW2 Tab scoreboard.

    The Tab screen always has a solid-colored panel across the top.
    We check that the left 30% of the top panel (y 2-6%) is a single
    uniform color (≤2 unique colors to allow minor compression artifacts).

    Returns True if valid, or a string describing why it was rejected.
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "failed to decode image"
    h, w = img.shape[:2]

    # Middle 30% of top panel (y 2-6%) — avoids tabs on left and map text on right
    strip = img[int(0.02 * h):int(0.06 * h), int(0.35 * w):int(0.65 * w)]
    unique = len(np.unique(strip.reshape(-1, 3), axis=0))
    if unique <= 2:
        return True
    return f"top strip has {unique} unique colors (max 2)"


def _find_overwatch_hwnd() -> int | None:
    """Find the Overwatch window handle, or None if not found."""
    result = {}

    def _enum_cb(hwnd, _):
        if not _user32.IsWindowVisible(hwnd):
            return True
        pid = ctypes.wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return True
        try:
            buf = ctypes.create_unicode_buffer(260)
            size = ctypes.wintypes.DWORD(260)
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                if buf.value.rsplit("\\", 1)[-1].lower() == "overwatch.exe":
                    result["hwnd"] = hwnd
                    return False
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        return True

    enum_func = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    _user32.EnumWindows(enum_func(_enum_cb), 0)
    return result.get("hwnd")


def _capture_window(hwnd: int) -> bytes | None:
    """Capture a window's content via PrintWindow (excludes overlays from other processes)."""
    import ctypes.wintypes

    rect = ctypes.wintypes.RECT()
    _user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right, rect.bottom
    if w <= 0 or h <= 0:
        return None

    # Create device contexts and bitmap
    _gdi32 = ctypes.windll.gdi32
    wnd_dc = _user32.GetDC(hwnd)
    mem_dc = _gdi32.CreateCompatibleDC(wnd_dc)
    bitmap = _gdi32.CreateCompatibleBitmap(wnd_dc, w, h)
    _gdi32.SelectObject(mem_dc, bitmap)

    # PrintWindow with PW_CLIENTONLY | PW_RENDERFULLCONTENT
    PW_CLIENTONLY = 0x1
    PW_RENDERFULLCONTENT = 0x2
    ok = _user32.PrintWindow(hwnd, mem_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT)

    if not ok:
        _gdi32.DeleteObject(bitmap)
        _gdi32.DeleteDC(mem_dc)
        _user32.ReleaseDC(hwnd, wnd_dc)
        return None

    # Read bitmap bits into a buffer
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0  # BI_RGB

    buf_size = w * h * 4
    buf = ctypes.create_string_buffer(buf_size)
    _gdi32.GetDIBits(mem_dc, bitmap, 0, h, buf, ctypes.byref(bmi), 0)

    _gdi32.DeleteObject(bitmap)
    _gdi32.DeleteDC(mem_dc)
    _user32.ReleaseDC(hwnd, wnd_dc)

    # Convert BGRA (Windows bitmap order) to RGB
    arr = np.frombuffer(buf.raw, dtype=np.uint8).reshape(h, w, 4)
    rgb = arr[:, :, 2::-1]  # BGRA -> RGB (reverse first 3 channels)
    img = Image.fromarray(rgb, "RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def capture_monitor() -> bytes:
    """Capture the Overwatch window (or full monitor as fallback) and return full-resolution PNG bytes."""
    hwnd = _find_overwatch_hwnd()
    if hwnd:
        result = _capture_window(hwnd)
        if result:
            return result
    # Fallback to full monitor capture
    with mss.mss() as sct:
        img = sct.grab(sct.monitors[MONITOR_INDEX])
        return mss.tools.to_png(img.rgb, img.size)
