"""Hero panel OCR and rank detection from tab screenshots.

Extracts hero-specific stats (featured stat + label/value pairs) and
match rank range from Overwatch 2 tab screen captures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_logger = logging.getLogger("overwatchlooker")

_MODELS_DIR = Path(__file__).parent / "models"
_ASSETS_DIR = Path(__file__).parent / "assets"
_RANK_ASSETS_DIR = _ASSETS_DIR / "ranks"
_HERO_ASSETS_DIR = _ASSETS_DIR / "heroes"

# Lazy-loaded models
_labels_model = None
_values_model = None


def _suppress_paddle_warnings():
    """Suppress noisy paddle/paddlex warnings during model loading."""
    import os
    import warnings
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["GLOG_minloglevel"] = "2"  # suppress C++ INFO/WARNING
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
    warnings.filterwarnings("ignore", message=".*ccache.*")
    for name in ("paddle", "paddlex", "paddleocr", "urllib3", "requests",
                 "chardet", "charset_normalizer"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _get_labels_model():
    global _labels_model
    if _labels_model is None:
        _suppress_paddle_warnings()
        from paddlex import create_model
        _labels_model = create_model(
            "PP-OCRv5_server_rec",
            model_dir=str(_MODELS_DIR / "panel_labels"),
        )
        _logger.info("Loaded panel labels OCR model")
    return _labels_model


def _get_values_model():
    global _values_model
    if _values_model is None:
        _suppress_paddle_warnings()
        from paddlex import create_model
        _values_model = create_model(
            "PP-OCRv5_server_rec",
            model_dir=str(_MODELS_DIR / "panel_values"),
        )
        _logger.info("Loaded panel values OCR model")
    return _values_model


def preload_models():
    """Load OCR models eagerly. Call at startup to avoid delay on first match."""
    _suppress_paddle_warnings()
    _logger.info("Preloading OCR models...")
    _get_values_model()
    _get_labels_model()
    _logger.info("OCR models ready")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HeroStat:
    label: str
    value: str
    is_featured: bool = False


@dataclass
class HeroPanelResult:
    stats: list[HeroStat]


@dataclass
class RankRange:
    min_rank: str  # e.g. "Bronze 2"
    max_rank: str  # e.g. "Gold 1"
    is_wide: bool


# ---------------------------------------------------------------------------
# Panel detection
# ---------------------------------------------------------------------------

def _detect_panel(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the hero stats panel rectangle via color-run scanning."""
    h, w = img.shape[:2]
    top_start = int(h * 0.12)
    right_start = int(w * 0.55)
    roi = img[top_start:, right_start:]
    rh, rw = roi.shape[:2]

    b, g, r = cv2.split(roi)
    mask = (
        (r >= 5) & (r <= 50) & (g >= 10) & (g <= 55) &
        (b >= 25) & (b <= 80) & (b > r) & (b > g)
    ).astype(np.uint8) * 255

    col_max_run = np.zeros(rw, dtype=int)
    for x in range(rw):
        col = mask[:, x]
        best = cur = 0
        for y in range(rh):
            if col[y] > 0:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        col_max_run[x] = best

    row_max_run = np.zeros(rh, dtype=int)
    for y in range(rh):
        row = mask[y, :]
        best = cur = 0
        for x in range(rw):
            if row[x] > 0:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        row_max_run[y] = best

    v_thresh = rh * 0.4
    v_cands = np.where(col_max_run > v_thresh)[0]
    if len(v_cands) == 0:
        return None
    panel_left, panel_right = v_cands[0], v_cands[-1]

    h_thresh = (panel_right - panel_left) * 0.5
    h_cands = np.where(row_max_run > h_thresh)[0]
    if len(h_cands) == 0:
        return None

    return (panel_left + right_start, h_cands[0] + top_start,
            panel_right - panel_left, h_cands[-1] - h_cands[0])


# ---------------------------------------------------------------------------
# Row detection + classification
# ---------------------------------------------------------------------------

def _find_text_rows(gray: np.ndarray) -> list[tuple[int, int]]:
    h, w = gray.shape
    bright_pct = (gray > 80).sum(axis=1).astype(float) / w * 100
    text_rows = bright_pct > 1.0
    blocks: list[tuple[int, int]] = []
    in_block = False
    start = 0
    for y in range(h):
        if text_rows[y] and not in_block:
            start = y
            in_block = True
        elif not text_rows[y] and in_block:
            blocks.append((start, y))
            in_block = False
    if in_block:
        blocks.append((start, h))
    merged: list[tuple[int, int]] = []
    for s, e in blocks:
        if merged and s - merged[-1][1] < 4:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= 6]


def _classify_rows(gray: np.ndarray, rows: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    classified = []
    for s, e in rows:
        strip = gray[s:e, :]
        bright = strip[strip > 80]
        avg = bright.mean() if len(bright) > 0 else 0
        if e - s > 100:
            t = "header"
        elif avg > 170:
            t = "value"
        else:
            t = "label"
        classified.append((s, e, t))
    return classified


# ---------------------------------------------------------------------------
# Crop + OCR helpers
# ---------------------------------------------------------------------------

def _crop_to_text(img_bgr: np.ndarray, threshold: int = 120, pad: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray.max(axis=0) > threshold)[0]
    rows = np.where(gray.max(axis=1) > threshold)[0]
    if len(cols) == 0 or len(rows) == 0:
        return img_bgr
    return img_bgr[
        max(0, rows[0] - pad):rows[-1] + pad,
        max(0, cols[0] - pad):cols[-1] + pad,
    ]


def _ocr_value(panel: np.ndarray, ys: int, ye: int) -> str:
    h = panel.shape[0]
    ys_pad = max(0, ys - 20)
    ye_pad = min(h, ye + 20)
    region = panel[ys_pad:ye_pad, :]
    cropped = _crop_to_text(region)
    results = list(_get_values_model().predict(cropped))
    if results:
        return results[0]["rec_text"].strip()
    return ""


def _ocr_label(panel: np.ndarray, ys: int, ye: int) -> str:
    region = panel[ys:ye, :]
    results = list(_get_labels_model().predict(region))
    if results:
        return results[0]["rec_text"].strip()
    return ""


# ---------------------------------------------------------------------------
# Featured stat
# ---------------------------------------------------------------------------

def _detect_featured_box(panel: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = panel.shape[:2]
    top = panel[:h // 4, w // 3:]
    b, g, r = cv2.split(top)
    mask = (
        (r >= 5) & (r <= 30) &
        (g >= 10) & (g <= 35) &
        (b >= 20) & (b <= 50) &
        (b > r)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(biggest)
    return (x + w // 3, y, cw, ch)


def _ocr_featured_stat(panel: np.ndarray, box: tuple[int, int, int, int]) -> HeroStat | None:
    bx, by, bw, bh = box
    featured = panel[by:by + bh, bx:bx + bw]
    h = featured.shape[0]
    split_y = int(h * 0.6)

    val_region = _crop_to_text(featured[:split_y, :])
    vr = list(_get_values_model().predict(val_region))
    val = vr[0]["rec_text"].strip() if vr else ""

    lbl_region = featured[split_y:, :]
    lr = list(_get_labels_model().predict(lbl_region))
    lbl = lr[0]["rec_text"].strip() if lr else ""

    if val or lbl:
        return HeroStat(label=lbl, value=val, is_featured=True)
    return None


# ---------------------------------------------------------------------------
# Public: read hero panel
# ---------------------------------------------------------------------------

def read_hero_panel(img: np.ndarray) -> HeroPanelResult | None:
    """Extract hero stats from a tab screenshot.

    Returns HeroPanelResult with featured stat + stat rows, or None if
    no panel is detected.
    """
    rect = _detect_panel(img)
    if rect is None:
        _logger.debug("No hero panel found in screenshot")
        return None

    px, py, pw, ph = rect
    panel = img[py:py + ph, px:px + pw]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    stats: list[HeroStat] = []

    # Featured stat
    box = _detect_featured_box(panel)
    if box:
        featured = _ocr_featured_stat(panel, box)
        if featured:
            stats.append(featured)

    # Regular stats — only rows below the header
    rows = _find_text_rows(gray)
    classified = _classify_rows(gray, rows)
    header_end = 0
    for s, e, t in classified:
        if t == "header":
            header_end = e
    raw_values = [(s, e) for s, e, t in classified if t == "value" and s > header_end]
    label_rows = [(s, e) for s, e, t in classified if t == "label" and s > header_end]

    # Merge consecutive value rows with small gaps (split glyphs like "2")
    value_rows: list[tuple[int, int]] = []
    for s, e in raw_values:
        if value_rows and s - value_rows[-1][1] < 10:
            value_rows[-1] = (value_rows[-1][0], e)
        else:
            value_rows.append((s, e))

    # Skip the first value (hero name row)
    if value_rows:
        value_rows = value_rows[1:]

    n = min(len(value_rows), len(label_rows))
    for i in range(n):
        val = _ocr_value(panel, *value_rows[i])
        lbl = _ocr_label(panel, *label_rows[i])
        stats.append(HeroStat(label=lbl, value=val))

    _logger.info(f"Hero panel OCR: {len(stats)} stats read")
    return HeroPanelResult(stats=stats)


# ---------------------------------------------------------------------------
# Rank detection
# ---------------------------------------------------------------------------

_rank_templates: dict[str, np.ndarray] | None = None
_div_templates: dict[str, np.ndarray] | None = None

_RANK_NAMES = ["bronze", "silver", "gold", "plat", "diamond", "master", "gm", "champion"]
_DIV_NAMES = ["division_1", "division_2", "division_3", "division_4", "division_5"]
_RANK_LABELS = {
    "bronze": "Bronze", "silver": "Silver", "gold": "Gold", "plat": "Platinum",
    "diamond": "Diamond", "master": "Master", "gm": "Grandmaster", "champion": "Champion",
}


def _load_rank_templates() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    global _rank_templates, _div_templates
    if _rank_templates is not None:
        return _rank_templates, _div_templates  # type: ignore

    _rank_templates = {}
    _div_templates = {}
    for name in _RANK_NAMES + _DIV_NAMES:
        path = _RANK_ASSETS_DIR / f"{name}_a.png"
        if path.exists():
            tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                if name.startswith("division_"):
                    _div_templates[name] = tmpl
                else:
                    _rank_templates[name] = tmpl

    _logger.info(f"Loaded {len(_rank_templates)} rank + {len(_div_templates)} division templates")
    return _rank_templates, _div_templates


def _find_all_matches(
    roi: np.ndarray, tmpl: np.ndarray,
    threshold: float = 0.88, min_dist: int = 50,
) -> list[tuple[float, int, int, int, int]]:
    """Find all template matches above threshold. Returns (score, center_x, y, w, h)."""
    th, tw = tmpl.shape
    all_hits: list[tuple[float, int, int, int, int]] = []
    for scale in np.arange(0.2, 1.2, 0.05):
        rw, rh = int(tw * scale), int(th * scale)
        if rw > roi.shape[1] or rh > roi.shape[0] or rw < 5 or rh < 5:
            continue
        resized = cv2.resize(tmpl, (rw, rh))
        result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= threshold)
        for y, x in zip(*locs):
            cx = x + rw // 2
            all_hits.append((result[y, x], cx, y, rw, rh))
    # NMS
    all_hits.sort(key=lambda h: -h[0])
    kept: list[tuple[float, int, int, int, int]] = []
    for hit in all_hits:
        too_close = any(abs(hit[1] - k[1]) < min_dist and abs(hit[2] - k[2]) < min_dist
                        for k in kept)
        if not too_close:
            kept.append(hit)
    return kept


def detect_rank_range(img: np.ndarray) -> RankRange | None:
    """Detect competitive rank range from a tab screenshot.

    Returns RankRange with min/max rank strings and wide match flag,
    or None if no rank icons are found (e.g. non-competitive match).
    """
    rank_tmpls, div_tmpls = _load_rank_templates()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    roi_x = int(w * 0.8)
    roi = gray[:int(h * 0.2), roi_x:]

    # Find rank icons
    rank_hits: list[tuple[str, float, int, int]] = []  # name, score, cx, y
    for name, tmpl in rank_tmpls.items():
        for score, cx, y, rw, rh in _find_all_matches(roi, tmpl):
            rank_hits.append((name, score, cx + roi_x, y))

    # Find division icons
    div_hits: list[tuple[int, float, int, int]] = []  # div_num, score, cx, y
    for name, tmpl in div_tmpls.items():
        div_num = int(name.split("_")[1])
        for score, cx, y, rw, rh in _find_all_matches(roi, tmpl):
            div_hits.append((div_num, score, cx + roi_x, y))

    if len(rank_hits) < 2:
        _logger.debug(f"Rank detection: found {len(rank_hits)} rank icons (need 2)")
        return None

    # Sort by x: left = min rank, right = max rank
    rank_hits.sort(key=lambda h: h[2])
    min_rank = rank_hits[0]
    max_rank = rank_hits[1]

    # Pair divisions by center-x proximity
    def best_div_for_cx(target_cx: int) -> int | None:
        close = [(d, s) for d, s, cx, y in div_hits if abs(cx - target_cx) < 60]
        if close:
            return max(close, key=lambda c: c[1])[0]
        return None

    min_div = best_div_for_cx(min_rank[2])
    max_div = best_div_for_cx(max_rank[2])

    min_str = f"{_RANK_LABELS[min_rank[0]]} {min_div}" if min_div else _RANK_LABELS[min_rank[0]]
    max_str = f"{_RANK_LABELS[max_rank[0]]} {max_div}" if max_div else _RANK_LABELS[max_rank[0]]

    # Wide match: check for yellow pixels left of the leftmost rank icon
    icon_x = min_rank[2]
    roi_wide = img[
        int(h * 0.05):int(h * 0.15),
        int(icon_x - w * 0.10):icon_x,
    ]
    b_w, g_w, r_w = cv2.split(roi_wide)
    yellow = (r_w > 180) & (g_w > 140) & (g_w < 220) & (b_w < 80)
    is_wide = yellow.sum() / yellow.size * 100 > 0.5

    _logger.info(f"Rank detected: {min_str} - {max_str} (wide={is_wide})")
    return RankRange(min_rank=min_str, max_rank=max_str, is_wide=is_wide)


# ---------------------------------------------------------------------------
# Hero ban detection
# ---------------------------------------------------------------------------

_hero_templates: dict[str, np.ndarray] | None = None


def _hero_to_filename(name: str) -> str:
    """Convert heroes.txt name to asset filename stem."""
    return name.replace(".", "").replace(":", "").replace(" ", "_")


def _load_hero_templates() -> dict[str, np.ndarray]:
    """Load hero portrait templates (inner 70%, BGR)."""
    global _hero_templates
    if _hero_templates is not None:
        return _hero_templates

    from overwatchlooker.heroes import ALL_HEROES

    _hero_templates = {}
    for hero in ALL_HEROES:
        fname = _hero_to_filename(hero)
        path = _HERO_ASSETS_DIR / f"{fname}.png"
        if not path.exists():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        th, tw = raw.shape[:2]
        # Crop inner 70% (remove 15% from each edge)
        m = int(th * 0.15)
        inner = raw[m:th - m, m:tw - m, :3]  # BGR only, no alpha
        _hero_templates[hero] = inner

    _logger.info(f"Loaded {len(_hero_templates)} hero portrait templates")
    return _hero_templates


def detect_hero_bans(img: np.ndarray, threshold: float = 0.8) -> list[str]:
    """Detect banned heroes from the top bar of a tab screenshot.

    Returns a list of hero names (Title Case, matching heroes.txt).
    """
    templates = _load_hero_templates()
    h, w = img.shape[:2]

    # Search top 6%, center portion (bans are in the top bar, center-left)
    roi = img[:int(h * 0.06), int(w * 0.15):int(w * 0.55)]
    rh, rw = roi.shape[:2]

    bans: list[tuple[str, float, int]] = []  # hero, score, x

    for hero, tmpl in templates.items():
        th, tw = tmpl.shape[:2]
        best_val = -1.0
        best_x = 0

        for scale in np.arange(0.2, 0.6, 0.02):
            sw, sh = int(tw * scale), int(th * scale)
            if sw > rw or sh > rh or sw < 10 or sh < 10:
                continue
            resized = cv2.resize(tmpl, (sw, sh))
            result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(result)
            if mv > best_val:
                best_val = mv
                best_x = ml[0]

        if best_val >= threshold:
            bans.append((hero, best_val, best_x))

    # Sort by x position (left to right)
    bans.sort(key=lambda b: b[2])
    result = [hero for hero, _, _ in bans]

    if result:
        _logger.info(f"Hero bans detected: {result}")
    return result
