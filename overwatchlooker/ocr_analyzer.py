"""EasyOCR-based scoreboard analyzer for Overwatch 2 Tab screens."""

import re
from dataclasses import dataclass, field

import cv2
import easyocr
import numpy as np

# --- Normalized region coordinates (fraction of image w, h) ---
REGION_MAP_HEADER = (0.70, 0.01, 0.99, 0.07)
REGION_YOUR_TEAM = (0.121, 0.180, 0.621, 0.500)
REGION_ENEMY_TEAM = (0.121, 0.545, 0.621, 0.865)
REGION_RESULT_TEXT = (0.200, 0.850, 0.800, 0.990)
REGION_HERO_FEATURED = (0.74, 0.13, 0.90, 0.25)  # big number + label (e.g. "9 PLAYERS SAVED")
REGION_HERO_STATS = (0.62, 0.24, 0.984, 0.78)    # hero name + stat lines (extra left margin for OCR)

# Column X-ranges (fraction of team crop width)
COL_RANGES = {
    "name": (0.10, 0.45),
    "EAD": (0.48, 0.67),   # combined E+A+D strip
    "DMG": (0.67, 0.78),
    "H": (0.78, 0.88),
    "MIT": (0.88, 1.00),
}

SUBTITLES = {"all-star", "medic", "knight", "ice breaker", "berserker", "strategist",
             "duelist", "flanker", "anchor", "guardian", "vanguard"}

NUM_PLAYERS_PER_TEAM = 5
ROLE_ORDER = ["TANK", "DPS", "DPS", "SUPPORT", "SUPPORT"]

_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


@dataclass
class PlayerStats:
    role: str = "UNKNOWN"
    name: str = ""
    elims: str = "-"
    assists: str = "-"
    deaths: str = "-"
    damage: str = "-"
    healing: str = "-"
    mitigation: str = "-"


@dataclass
class HeroStats:
    hero_name: str = ""
    featured_stat: str = ""       # e.g. "Players Saved: 9"
    stats: list[str] = field(default_factory=list)  # e.g. ["Weapon Accuracy: 54%", ...]


@dataclass
class MatchData:
    map_name: str = ""
    time: str = ""
    mode: str = ""
    result: str = "UNKNOWN"
    your_team: list[PlayerStats] = field(default_factory=list)
    enemy_team: list[PlayerStats] = field(default_factory=list)
    hero_stats: HeroStats | None = None


def _crop_region(img: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = int(region[0] * w), int(region[1] * h)
    x2, y2 = int(region[2] * w), int(region[3] * h)
    return img[y1:y2, x1:x2]


def _extract_map_header(img: np.ndarray) -> tuple[str, str, str]:
    """Extract (mode, map_name, time) from the top-right header."""
    crop = _crop_region(img, REGION_MAP_HEADER)
    upscaled = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    reader = _get_reader()
    # Use detail=1 to get positions, sort by X for correct reading order
    results = reader.readtext(upscaled, detail=1, paragraph=False)
    sorted_by_x = sorted(results, key=lambda r: sum(pt[0] for pt in r[0]) / 4)
    raw = " ".join(r[1] for r in sorted_by_x).upper()

    # Fix split time: "1 0:00" -> "10:00", "1 2.24" -> "12.24"
    # Merge lone digit before a digit-separator-XX pattern (: or . as separator)
    raw = re.sub(r"(\d)\s+(\d[:.]\d{2})", r"\1\2", raw)

    mode, map_name, time_str = "", "", ""
    modes = ["CONTROL", "ESCORT", "PUSH", "HYBRID", "CLASH", "FLASHPOINT"]

    def _title_case(s: str) -> str:
        """Title case that handles apostrophes correctly (King's Row, not King'S Row)."""
        return re.sub(r"'\w", lambda m: m.group(0).lower(), s.title())

    # Try full pattern: MODE | MAP TIME: MM:SS
    m = re.search(
        r"(" + "|".join(modes) + r")\s*[|\s]\s*(.+?)\s*TIME\s*[:'.;]?\s*(\d{1,2})\s*[:.*]?\s*(\d{2})",
        raw,
    )
    if m:
        mode = m.group(1).title()
        map_name = _title_case(m.group(2).strip(" |"))
        time_str = f"{m.group(3)}:{m.group(4)}"
    else:
        # Mode might be an icon (no text) — try: MAP TIME: MM:SS
        m2 = re.search(
            r"(.+?)\s*TIME\s*[:'.;]?\s*(\d{1,2})\s*[:.*]?\s*(\d{2})",
            raw,
        )
        if m2:
            candidate_map = m2.group(1).strip(" |")
            # Strip mode text if present at the start
            for mode_name in modes:
                if candidate_map.startswith(mode_name):
                    mode = mode_name.title()
                    candidate_map = candidate_map[len(mode_name):].strip(" |")
                    break
            map_name = _title_case(candidate_map)
            time_str = f"{m2.group(2)}:{m2.group(3)}"
        else:
            t = re.search(r"(\d{1,2})\s*[:.*]?\s*(\d{2})", raw)
            if t:
                time_str = f"{t.group(1)}:{t.group(2)}"

    return mode, map_name, time_str


def _extract_result(img: np.ndarray) -> str:
    """Detect VICTORY/DEFEAT from ATHENA subtitle at bottom center."""
    crop = _crop_region(img, REGION_RESULT_TEXT)
    upscaled = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    reader = _get_reader()
    results = reader.readtext(upscaled, detail=0, paragraph=True)
    raw = " ".join(results).upper()

    if "VICTORY" in raw:
        return "VICTORY"
    elif "DEFEAT" in raw:
        return "DEFEAT"
    elif "DRAW" in raw:
        return "DRAW"
    return "UNKNOWN"


def _clean_stat(raw: str) -> str:
    if not raw or raw.isspace():
        return "-"
    cleaned = raw.replace(" ", "").replace(",,", ",")
    cleaned = re.sub(r"(\d)\.(\d)", r"\1,\2", cleaned)
    cleaned = cleaned.replace("O", "0").replace("o", "0").replace("l", "1")
    digits = re.sub(r"[^\d,]", "", cleaned)
    if not digits:
        return "-"
    # Extract the first valid number: either "X,XXX" or "XX,XXX" or plain digits up to 3
    m = re.match(r"(\d{1,3},\d{3}|\d{1,3})", digits)
    return m.group(1) if m else digits


def _is_subtitle(text: str) -> bool:
    return text.lower().strip() in SUBTITLES


def _cell_has_text(cell: np.ndarray) -> bool:
    """Check if a cell region has enough bright pixels to contain text.
    OW2 renders '0' values in grey (~150-178 brightness), not white."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > 120)
    # A single "0" digit is about 0.5-1% of the cell area
    return bright_pixels > (cell.shape[0] * cell.shape[1] * 0.005)


def _cell_is_dim(cell: np.ndarray) -> bool:
    """Check if cell text is dim grey (OW2 renders '0' values dimmer than non-zero).
    Uses HSV to isolate actual text (low saturation white/grey) from colored UI
    elements like the highlighted first-player row background."""
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    center = hsv[:, int(w * 0.2):int(w * 0.8)]
    # Text pixels: low saturation (white/grey, not colored UI), bright enough to be text
    text_mask = (center[:, :, 1] < 35) & (center[:, :, 2] > 140)
    text_values = center[:, :, 2][text_mask]
    if len(text_values) == 0:
        return True
    # Non-zero stats are bright white (V>200); zero stats are dim grey (V~150-178)
    bright_ratio = float(np.sum(text_values > 200)) / len(text_values)
    return bright_ratio < 0.3


def _ocr_small_cell(cell: np.ndarray) -> tuple[str, float]:
    """Focused OCR on a small cell with aggressive preprocessing.
    Returns (cleaned_value, confidence)."""
    scale = max(3.0, min(10.0, 400.0 / max(cell.shape[0], 1)))
    up = cv2.resize(cell, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    # Dilate to thicken thin strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    reader = _get_reader()
    results = reader.readtext(dilated, detail=1, allowlist="0123456789,.")
    if not results:
        return ("", 0.0)
    best = max(results, key=lambda r: r[2])
    raw = best[1].strip()
    conf = float(best[2])
    return (_clean_stat(raw) if raw else "", conf)


def _extract_team_stats(img: np.ndarray, team_region: tuple) -> list[PlayerStats]:
    """Hybrid: batch OCR for names + wide stats, strip OCR for E/A/D."""
    team_crop = _crop_region(img, team_region)
    h, w = team_crop.shape[:2]
    row_height = h / NUM_PLAYERS_PER_TEAM

    # --- Batch OCR for names and wide columns ---
    # Scale inversely with resolution: 3x for 720p, ~1.5x for 4K
    scale = max(1.5, min(3.0, 2200.0 / h))
    upscaled = cv2.resize(team_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    reader = _get_reader()
    results = reader.readtext(upscaled, detail=1)

    # (x_frac, text, conf, y_frac_in_row) — y_frac_in_row: 0=top, 1=bottom of row
    rows: list[list[tuple[float, str, float, float]]] = [[] for _ in range(NUM_PLAYERS_PER_TEAM)]
    for bbox, text, conf in results:
        y_center = sum(pt[1] for pt in bbox) / 4
        x_center = sum(pt[0] for pt in bbox) / 4
        x_frac = x_center / (w * scale)
        row_idx = min(int(y_center / (row_height * scale)), NUM_PLAYERS_PER_TEAM - 1)
        y_in_row = (y_center - row_idx * row_height * scale) / (row_height * scale)
        rows[row_idx].append((x_frac, text, conf, y_in_row))

    wide_stat_fields = {"DMG": "damage", "H": "healing", "MIT": "mitigation"}
    players = []

    for i, row_detections in enumerate(rows):
        player = PlayerStats(role=ROLE_ORDER[i])

        # Track best (highest confidence) detection per column
        col_best: dict[str, tuple[str, float]] = {}

        # Collect all name-column candidates with their Y position in the row
        name_candidates: list[tuple[str, float, float]] = []  # (text, conf, y_in_row)
        for x_frac, text, conf, y_in_row in row_detections:
            if COL_RANGES["name"][0] <= x_frac <= COL_RANGES["name"][1]:
                stripped = text.strip()
                if stripped:
                    name_candidates.append((stripped, conf, y_in_row))
                continue
            # Skip low-confidence detections
            if conf < 0.25:
                continue
            for col_key, field_name in wide_stat_fields.items():
                lo, hi = COL_RANGES[col_key]
                if lo <= x_frac <= hi:
                    prev = col_best.get(col_key)
                    if prev is None or conf > prev[1]:
                        col_best[col_key] = (text, conf)
                    break

        # Pick the best name using heuristics:
        # - BattleTags never have spaces; subtitles always do
        # - Subtitles appear in the bottom half of the row
        # - Rank badges are short digit-only strings
        # Priority: no-space non-digit top > no-space non-digit > any non-digit top > fallback
        def _name_score(n: str, c: float, y: float) -> tuple[int, float]:
            is_digit = n.isdigit()
            has_space = " " in n
            is_known_sub = _is_subtitle(n)
            top = y < 0.55
            if is_known_sub or is_digit:
                return (0, c)
            if not has_space and top:
                return (4, c)
            if not has_space:
                return (3, c)
            if top:
                return (2, c)
            return (1, c)

        if name_candidates:
            best = max(name_candidates, key=lambda x: _name_score(*x))
            player.name = best[0]

        for col_key, field_name in wide_stat_fields.items():
            if col_key in col_best:
                setattr(player, field_name, _clean_stat(col_best[col_key][0]))

        y_start = int(i * row_height)
        y_end = int((i + 1) * row_height)
        row_crop = team_crop[y_start:y_end, :]

        # --- Fill missing or validate wide stats ---
        for col_key, field_name in wide_stat_fields.items():
            lo, hi = COL_RANGES[col_key]
            cell = row_crop[:, int(lo * w):int(hi * w)]
            if not _cell_has_text(cell):
                continue
            current = getattr(player, field_name)
            if _cell_is_dim(cell):
                # Dim grey text = OW2 zero value, regardless of what OCR said
                setattr(player, field_name, "0")
            elif current == "-":
                val, conf = _ocr_small_cell(cell)
                if conf >= 0.15 and val:
                    setattr(player, field_name, val)
                else:
                    setattr(player, field_name, "0")
            elif len(current) == 1 and current != "0":
                # Single digit from batch OCR — validate with focused OCR
                val, conf = _ocr_small_cell(cell)
                if conf < 0.15:
                    setattr(player, field_name, "0")

        # --- E/A/D strip OCR per row ---
        ead_lo, ead_hi = COL_RANGES["EAD"]
        ead_crop = row_crop[:, int(ead_lo * w):int(ead_hi * w)]

        # If the entire E/A/D strip is dim, all three are "0"
        if _cell_has_text(ead_crop) and _cell_is_dim(ead_crop):
            player.elims, player.assists, player.deaths = "0", "0", "0"
        else:
            ead_scale = max(2.0, min(8.0, 600.0 / ead_crop.shape[0]))
            ead_up = cv2.resize(ead_crop, None, fx=ead_scale, fy=ead_scale, interpolation=cv2.INTER_CUBIC)
            ead_results = reader.readtext(
                ead_up, detail=1, allowlist="0123456789", paragraph=False
            )
            # Sort by X position, filter out artifacts at strip edges
            strip_w = ead_up.shape[1]
            ead_sorted = sorted(ead_results, key=lambda r: sum(pt[0] for pt in r[0]) / 4)
            ead_vals = []
            for r in ead_sorted:
                x_center = sum(pt[0] for pt in r[0]) / 4
                if x_center < strip_w * 0.03 or x_center > strip_w * 0.97:
                    continue
                if r[1].strip():
                    ead_vals.append(r[1].strip())

            if len(ead_vals) >= 1:
                player.elims = ead_vals[0]
            if len(ead_vals) >= 2:
                player.assists = ead_vals[1]
            if len(ead_vals) >= 3:
                player.deaths = ead_vals[2]
            # Pad missing slots with "0" if strip has content
            if _cell_has_text(ead_crop) and len(ead_vals) < 3:
                if player.elims == "-":
                    player.elims = "0"
                if player.assists == "-":
                    player.assists = "0"
                if player.deaths == "-":
                    player.deaths = "0"

        players.append(player)

    return players


def _extract_hero_stats(img: np.ndarray) -> HeroStats | None:
    """Extract hero-specific stats from the right panel."""
    reader = _get_reader()
    hero = HeroStats()

    # --- Featured stat (big number + label, e.g. "9 PLAYERS SAVED") ---
    feat_crop = _crop_region(img, REGION_HERO_FEATURED)
    feat_up = cv2.resize(feat_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    h_feat = feat_up.shape[0]
    # Top half: the big number
    feat_num = feat_up[:h_feat // 2, :]
    num_results = reader.readtext(feat_num, detail=0, allowlist="0123456789,.", paragraph=True)
    # Bottom half: the label
    feat_label = feat_up[h_feat // 2:, :]
    label_results = reader.readtext(feat_label, detail=0, paragraph=True)
    feat_number = num_results[0].strip() if num_results else ""
    feat_label_text = " ".join(label_results).strip() if label_results else ""
    if feat_label_text and feat_number:
        hero.featured_stat = f"{feat_label_text}: {feat_number}"
    elif feat_label_text:
        hero.featured_stat = f"{feat_label_text}: 0"
    elif feat_number:
        hero.featured_stat = feat_number

    # --- Hero name + stat lines ---
    stats_crop = _crop_region(img, REGION_HERO_STATS)
    stats_up = cv2.resize(stats_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(stats_up, detail=1, paragraph=False)

    # Sort by Y position, filter low-confidence noise (icons misread as digits)
    sorted_results = sorted(results, key=lambda r: sum(pt[1] for pt in r[0]) / 4)
    sorted_results = [r for r in sorted_results if r[2] > 0.15 or len(r[1].strip()) > 2]

    if not sorted_results:
        return hero

    # First item is the hero name
    hero.hero_name = sorted_results[0][1].strip()

    # Remaining items alternate: value (has digits/%), label (all-caps words)
    # Group into value+label pairs
    items = sorted_results[1:]
    pending_value = None
    for _, text, conf in items:
        text = text.strip()
        if not text:
            continue
        # A value contains at least one digit or %
        is_value = bool(re.search(r"[\d%]", text))
        if is_value:
            # Clean up common OCR artifacts in values
            text = text.rstrip("/").strip()
            # Bare "%" means OCR missed the "0" before it
            if text == "%":
                text = "0%"
            # If we had a pending value without a label, save it alone
            if pending_value is not None:
                hero.stats.append(pending_value)
            pending_value = text
        else:
            # This is a label
            if pending_value is not None:
                hero.stats.append(f"{text}: {pending_value}")
                pending_value = None
            else:
                # Orphan label — value was likely "0" that OCR missed
                hero.stats.append(f"{text}: 0")

    # Don't forget the last pending value
    if pending_value is not None:
        hero.stats.append(pending_value)

    return hero if hero.hero_name else None


def _verify_scoreboard(img: np.ndarray) -> bool:
    """Check if the image looks like an OW2 scoreboard (blue top, red bottom)."""
    your_crop = _crop_region(img, REGION_YOUR_TEAM)
    enemy_crop = _crop_region(img, REGION_ENEMY_TEAM)

    your_hsv = cv2.cvtColor(your_crop, cv2.COLOR_BGR2HSV)
    enemy_hsv = cv2.cvtColor(enemy_crop, cv2.COLOR_BGR2HSV)

    your_avg_hue = np.mean(your_hsv[:, :, 0])
    enemy_avg_hue = np.mean(enemy_hsv[:, :, 0])

    blue_ok = 70 < your_avg_hue < 140
    red_ok = enemy_avg_hue < 25 or enemy_avg_hue > 135
    return blue_ok and red_ok


def _format_output(data: MatchData) -> str:
    lines = [
        f"MAP: {data.map_name}",
        f"TIME: {data.time}",
        f"MODE: {data.mode}",
        f"RESULT: {data.result}",
        "",
        "=== YOUR TEAM ===",
        "Role | Player | E | A | D | DMG | H | MIT",
    ]
    for p in data.your_team:
        lines.append(
            f"{p.role} | {p.name} | {p.elims} | {p.assists} | {p.deaths} | "
            f"{p.damage} | {p.healing} | {p.mitigation}"
        )
    lines.extend(["", "=== ENEMY TEAM ===", "Role | Player | E | A | D | DMG | H | MIT"])
    for p in data.enemy_team:
        lines.append(
            f"{p.role} | {p.name} | {p.elims} | {p.assists} | {p.deaths} | "
            f"{p.damage} | {p.healing} | {p.mitigation}"
        )

    if data.hero_stats:
        hs = data.hero_stats
        lines.append("")
        lines.append("HERO STATS:")
        stat_str = "; ".join(hs.stats)
        if hs.featured_stat:
            lines.append(f"{hs.hero_name} - {hs.featured_stat}; {stat_str}")
        else:
            lines.append(f"{hs.hero_name} - {stat_str}")

    return "\n".join(lines)


def analyze_screenshot(png_bytes: bytes, audio_result: str | None = None) -> str:
    """Analyze an OW2 scoreboard screenshot using EasyOCR."""
    from overwatchlooker.display import print_status

    img = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "NOT_OW2_TAB: Could not decode image."

    print_status(f"Image size: {img.shape[1]}x{img.shape[0]}")

    if not _verify_scoreboard(img):
        return "NOT_OW2_TAB: This does not appear to be an Overwatch 2 scoreboard screenshot."

    print_status("Extracting map header...")
    mode, map_name, time_str = _extract_map_header(img)

    print_status("Extracting result...")
    result = _extract_result(img)
    if result == "UNKNOWN" and audio_result:
        result = audio_result

    print_status("Extracting your team stats...")
    your_team = _extract_team_stats(img, REGION_YOUR_TEAM)

    print_status("Extracting enemy team stats...")
    enemy_team = _extract_team_stats(img, REGION_ENEMY_TEAM)

    print_status("Extracting hero stats...")
    hero_stats = _extract_hero_stats(img)

    return _format_output(
        MatchData(
            map_name=map_name,
            time=time_str,
            mode=mode,
            result=result,
            your_team=your_team,
            enemy_team=enemy_team,
            hero_stats=hero_stats,
        )
    )
