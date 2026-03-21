import ctypes
import ctypes.wintypes
import io
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import mss
import mss.tools
import numpy as np
from PIL import Image

from pytesseract_api import image_to_string as _tess_image_to_string, TessPageSegMode

# Legacy constants kept for backward compat with unused functions
_MONITOR_INDEX = 1
_SCREENSHOT_MAX_AGE_SECONDS = 120.0

# Tesseract C API paths (ctypes, no subprocess)
_TESS_LIB = r"C:\Program Files\Tesseract-OCR\libtesseract-5.dll"
_TESS_DATA = r"C:\Program Files\Tesseract-OCR\tessdata"

_logger = logging.getLogger("overwatchlooker")
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
}


def _get_pictures_folder() -> Path:
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_MYPICTURES, None, 0, buf)
    return Path(buf.value)


def get_screenshots_dir() -> Path:
    folder = _get_pictures_folder() / "OverwatchLooker"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_screenshot(png_bytes: bytes, tick: int | None = None) -> Path:
    """Save screenshot to Pictures/OverwatchLooker with a timestamp filename.

    When tick is provided, it's appended to avoid collisions during fast replay.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{timestamp}_t{tick}" if tick is not None else timestamp
    path = get_screenshots_dir() / f"{name}.png"
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
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
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
    if age > _SCREENSHOT_MAX_AGE_SECONDS:
        return None
    return latest


def is_ow2_tab_screen_bgr(frame: np.ndarray) -> tuple[bool, str]:
    """Fast check whether a BGR frame is an OW2 Tab scoreboard.

    Works directly on a BGR numpy array, avoiding PNG encode/decode per frame.
    Returns (True, "") if valid, or (False, reason) if rejected.
    """
    if frame is None or frame.size == 0:
        return False, "empty frame"
    h, w = frame.shape[:2]
    strip = frame[int(0.02 * h):int(0.06 * h), int(0.35 * w):int(0.65 * w)]
    unique = len(np.unique(strip.reshape(-1, 3), axis=0))
    if unique <= 2:
        return True, ""
    return False, f"top strip has {unique} unique colors (max 2)"


def is_ow2_tab_screen(png_bytes: bytes) -> tuple[bool, str]:
    """Fast check whether a screenshot is an OW2 Tab scoreboard.

    The Tab screen always has a solid-colored panel across the top.
    We check that the left 30% of the top panel (y 2-6%) is a single
    uniform color (≤2 unique colors to allow minor compression artifacts).

    Returns (True, "") if valid, or (False, reason) if rejected.
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False, "failed to decode image"
    return is_ow2_tab_screen_bgr(img)


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


# Hero stats panel region (fraction of image: x1, y1, x2, y2)
HERO_PANEL_REGION = (0.60, 0.12, 0.91, 0.85)



def has_hero_panel(png_bytes: bytes) -> bool:
    """Check if the screenshot has a hero stats panel on the right side.

    The hero panel has white text (stat labels/values) on a dark background.
    Returns True if the white text ratio exceeds the threshold.
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    x1, y1 = int(HERO_PANEL_REGION[0] * w), int(HERO_PANEL_REGION[1] * h)
    x2, y2 = int(HERO_PANEL_REGION[2] * w), int(HERO_PANEL_REGION[3] * h)
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    # Check the stats area (center of crop, avoids scoreboard columns on left)
    strip = crop[int(ch * 0.25):int(ch * 0.75), int(cw * 0.15):int(cw * 0.75)]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    white_text = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30)
    white_ratio = np.sum(white_text) / (strip.shape[0] * strip.shape[1])
    return white_ratio >= 0.008


def crop_hero_panel(png_bytes: bytes) -> bytes:
    """Crop the hero stats panel region and return as PNG bytes."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None
    h, w = img.shape[:2]
    x1, y1 = int(HERO_PANEL_REGION[0] * w), int(HERO_PANEL_REGION[1] * h)
    x2, y2 = int(HERO_PANEL_REGION[2] * w), int(HERO_PANEL_REGION[3] * h)
    crop = img[y1:y2, x1:x2]
    _, buf = cv2.imencode(".png", crop)
    return buf.tobytes()


def ocr_hero_name(crop_png_bytes: bytes) -> str:
    """OCR the hero name from a hero panel crop.

    The hero name appears below the hero portrait (~y 0.28-0.38 in the crop),
    as an all-caps word like "REINHARDT", "JUNO", "MOIRA".
    Returns the canonical hero name, or empty string if OCR fails.
    """
    import re

    arr = np.frombuffer(crop_png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""

    h, w = img.shape[:2]
    # Crop hero name row, left 55% only (excludes role icon on the right)
    name_region = img[int(h * 0.28):int(h * 0.38), :int(w * 0.55)]

    # Binarize: keep white/bright text with relaxed thresholds
    hsv = cv2.cvtColor(name_region, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 2] > 160) & (hsv[:, :, 1] < 50)
    binary = np.zeros(name_region.shape[:2], dtype=np.uint8)
    binary[white_mask] = 255

    # Dilate to thicken thin strokes, then upscale
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)  # type: ignore[assignment]
    binary = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)  # type: ignore[assignment]
    _, binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)  # type: ignore[assignment]

    try:
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        try:
            text = _tess_image_to_string(
                binary, psm=TessPageSegMode.PSM_SINGLE_LINE,
                lib_path=_TESS_LIB, tessdata_path=_TESS_DATA,
            ).strip()
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull)
    except Exception as e:
        _logger.warning(f"Hero name OCR failed: {e}")
        return ""

    # Clean: keep only alpha
    text = re.sub(r"[^a-zA-Z]", "", text).strip()
    if not text or len(text) < 3:
        return ""

    # Fuzzy-match against known hero list
    from overwatchlooker.heroes import match_hero_name
    matched = match_hero_name(text)
    if matched:
        return matched

    _logger.warning(f"Hero name OCR '{text}' didn't match any known hero")
    return ""


def capture_monitor() -> bytes:
    """Capture the Overwatch window (or full monitor as fallback) and return full-resolution PNG bytes."""
    hwnd = _find_overwatch_hwnd()
    if hwnd:
        result = _capture_window(hwnd)
        if result:
            return result
    # Fallback to full monitor capture
    with mss.mss() as sct:
        img = sct.grab(sct.monitors[_MONITOR_INDEX])
        png = mss.tools.to_png(img.rgb, img.size)
        assert png is not None
        return png
