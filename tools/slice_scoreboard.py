"""OW2 Scoreboard OCR — Stage 1: Row slicing + cell extraction + OCR.

Detects ally and enemy scoreboards from a Tab screenshot, slices into
individual player rows, identifies text groups (filtering perk icons),
and runs Tesseract OCR on each cell.

Usage:
    uv run python tools/slice_scoreboard.py screenshot.png [--debug DIR]

Pipeline:
    1. Color mask for ally blue / enemy red backgrounds
    2. Largest contour = scoreboard bounding box
    3. Count rows via white text density peaks in right half
    4. Equal-slice into N rows
    5. Trim portrait from left (scan for first all-background column)
    6. Detect & filter perk icons via HoughCircles + black/white ratio
    7. Group remaining white text blobs by horizontal proximity
    8. Skip ult charge (ally only), OCR each group
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Color definitions (RGB)
# ---------------------------------------------------------------------------

ALLY_BG_COLORS = [
    (2, 78, 108),    # regular ally row
    (1, 126, 171),   # self-row (highlighted)
]

ENEMY_BG_COLORS = [
    (100, 15, 29),   # regular enemy row
]


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------

def find_scoreboard_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find the largest contour in a color mask — the scoreboard bounding box."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No scoreboard found in mask")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return cv2.boundingRect(contours[0])


def count_player_rows(img_bgr: np.ndarray, bx: int, by: int, bw: int, bh: int) -> int:
    """Count player rows by detecting white text density peaks in the right half."""
    roi = img_bgr[by:by + bh, bx + bw // 2:bx + bw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    density = np.mean((gray > 140).astype(float), axis=1)
    smoothed = uniform_filter1d(density, size=8)
    peaks, _ = find_peaks(smoothed, height=0.01, distance=bh // 10)
    return len(peaks)


def find_portrait_end(row_bgr: np.ndarray, team_colors: list[tuple],
                      tolerance: float = 25) -> int:
    """Find where the portrait ends by scanning for the first column where
    all pixels match the team background color."""
    rgb = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    y_start, y_end = int(h * 0.2), int(h * 0.8)
    strip = rgb[y_start:y_end, :, :].astype(float)

    best = 0
    for tc in team_colors:
        diff = np.sqrt(np.sum((strip - np.array(tc, dtype=float)) ** 2, axis=2))
        all_match = np.all(diff < tolerance, axis=0)
        for x in range(w):
            if all_match[x]:
                best = max(best, x)
                break
    return best


def find_large_circles(row_bgr: np.ndarray) -> list[tuple[int, int, int]]:
    """Detect all large circles >50% row height (ult charge, perk icons).

    These are filtered out of text groups since they aren't OCR targets.
    """
    row_h = row_bgr.shape[0]
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=100, param2=40, minRadius=25, maxRadius=60,
    )
    result = []
    if circles is not None:
        for x, y, rad in np.round(circles[0]).astype(int):
            if (rad * 2) / row_h >= 0.5:
                result.append((x, y, rad))
    return result


def find_text_groups(row_bgr: np.ndarray,
                     circles: list[tuple[int, int, int]],
                     ) -> list[tuple[int, int, int, int]]:
    """Find white text bounding boxes, merge nearby ones, filter out circles.

    Circles (ult charge, perks) are masked out BEFORE merging to prevent
    them from merging with adjacent text (e.g. ult "33" + player name).

    Returns list of (x, y, w, h) sorted left to right.
    """
    row_h, row_w = row_bgr.shape[:2]
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    white = (gray > 140).astype(np.uint8) * 255

    # Erase circle regions from the white mask before finding contours
    for cx, cy, cr in circles:
        cv2.circle(white, (cx, cy), cr + 5, 0, -1)  # +5px margin

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted(
        [(x, y, w, h) for c in contours
         for x, y, w, h in [cv2.boundingRect(c)] if w > 3 and h > 3],
        key=lambda b: b[0],
    )
    if not boxes:
        return []

    # Merge boxes closer than 2% of row width
    merge_gap = row_w * 0.02
    groups: list[list[int]] = []
    cur = list(boxes[0])
    for bx, by, bw, bh in boxes[1:]:
        if bx - (cur[0] + cur[2]) < merge_gap:
            nx = min(cur[0], bx)
            ny = min(cur[1], by)
            cur = [nx, ny, max(cur[0] + cur[2], bx + bw) - nx,
                   max(cur[1] + cur[3], by + bh) - ny]
        else:
            groups.append(cur)
            cur = [bx, by, bw, bh]
    groups.append(cur)

    return [tuple(g) for g in groups]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

_OW2_TESSDATA = str(Path(__file__).parent.parent / "training_data" / "output")


def ocr_crop(crop_bgr: np.ndarray, lang: str = "ow2",
             tessdata: str = _OW2_TESSDATA, psm: int = 7) -> str:
    """Run Tesseract on a BGR image crop."""
    tmp = Path(tempfile.mktemp(suffix=".png"))
    cv2.imwrite(str(tmp), crop_bgr)
    try:
        result = subprocess.run(
            ["tesseract", str(tmp), "stdout", "-l", lang, "--psm", str(psm)],
            capture_output=True, text=True,
            env={**os.environ, "TESSDATA_PREFIX": tessdata},
        )
        return result.stdout.strip()
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def make_color_mask(img_bgr: np.ndarray, team: str) -> np.ndarray:
    """Create a binary mask for ally blue or enemy red pixels."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    if team == "ally":
        return ((r < 30) & (g > 40) & (g < 160) &
                (b > 60) & (b < 210)).astype(np.uint8) * 255
    else:
        return ((r > 70) & (r < 130) & (g < 30) &
                (b > 10) & (b < 50)).astype(np.uint8) * 255


def _extract_rows(img_bgr: np.ndarray, mask: np.ndarray,
                  ) -> tuple[int, int, int, int, int, int]:
    """Find scoreboard rect, count rows, return (bx, by, bw, bh, n_rows, row_h)."""
    bx, by, bw, bh = find_scoreboard_rect(mask)
    n_rows = count_player_rows(img_bgr, bx, by, bw, bh)
    row_h = bh // n_rows
    return bx, by, bw, bh, n_rows, row_h


def _get_portrait_cut(img_bgr: np.ndarray, bx: int, by: int, bw: int,
                      row_h: int, n_rows: int, bh: int,
                      team_colors: list[tuple]) -> int:
    """Compute median portrait cut-x across all rows of a team."""
    cuts = []
    for i in range(n_rows):
        y1 = i * row_h
        y2 = (i + 1) * row_h if i < n_rows - 1 else bh
        row_img = img_bgr[by + y1:by + y2, bx:bx + bw]
        cut = find_portrait_end(row_img, team_colors)
        if cut > 0:
            cuts.append(cut)
    return int(np.median(cuts)) if cuts else 0


def process_scoreboard(img_bgr: np.ndarray,
                       debug_dir: Path | None = None,
                       ) -> tuple[list[dict], list[dict]]:
    """Extract player data from both teams.

    Processes enemy first to get a clean portrait cut position (no ult
    charge effects), then applies the same cut to ally rows.

    Returns (ally_players, enemy_players).
    """
    ally_mask = make_color_mask(img_bgr, "ally")
    enemy_mask = make_color_mask(img_bgr, "enemy")

    if debug_dir:
        cv2.imwrite(str(debug_dir / "ally_mask.png"), ally_mask)
        cv2.imwrite(str(debug_dir / "enemy_mask.png"), enemy_mask)

    # Enemy first — clean portrait detection (no ult effects)
    e_bx, e_by, e_bw, e_bh, e_n, e_row_h = _extract_rows(img_bgr, enemy_mask)
    portrait_cut = _get_portrait_cut(
        img_bgr, e_bx, e_by, e_bw, e_row_h, e_n, e_bh, ENEMY_BG_COLORS)

    # Ally
    a_bx, a_by, a_bw, a_bh, a_n, a_row_h = _extract_rows(img_bgr, ally_mask)

    stat_labels = ["E", "A", "D", "DMG", "H", "MIT"]

    def _process_team(bx, by, bw, bh, n_rows, row_h, team, is_ally):
        players = []
        for i in range(n_rows):
            y1 = i * row_h
            y2 = (i + 1) * row_h if i < n_rows - 1 else bh
            row_img = img_bgr[by + y1:by + y2, bx:bx + bw]

            # Trim portrait (use enemy-derived cut for both teams)
            trimmed = row_img[:, portrait_cut:]

            # For ally, cut the ult charge area (slightly less than full square)
            if is_ally:
                trimmed = trimmed[:, int(row_h * 0.95):]

            # Detect circles (perks) and text groups
            circles = find_large_circles(trimmed)
            groups = find_text_groups(trimmed, circles)

            # OCR each group with padding
            th, tw = trimmed.shape[:2]
            values = []
            for idx, (gx, gy, gw, gh) in enumerate(groups):
                pad = max(4, gh // 3)
                x1 = max(0, gx - pad)
                y1_ = max(0, gy - pad)
                x2 = min(tw, gx + gw + pad)
                y2_ = min(th, gy + gh + pad)
                crop = trimmed[y1_:y2_, x1:x2]

                if idx == 0:
                    # Name group: may have title below, use PSM 6 + take first line
                    text = ocr_crop(crop, psm=6)
                    text = text.split("\n")[0].strip()
                else:
                    text = ocr_crop(crop)
                values.append(text)

            player = {"name": values[0] if values else ""}
            for j, label in enumerate(stat_labels):
                player[label] = values[j + 1] if j + 1 < len(values) else ""
            players.append(player)

            if debug_dir:
                cv2.imwrite(str(debug_dir / f"{team}_row_{i:02d}.png"), row_img)
                cv2.imwrite(str(debug_dir / f"{team}_row_{i:02d}_trimmed.png"), trimmed)

        return players

    enemy = _process_team(e_bx, e_by, e_bw, e_bh, e_n, e_row_h, "enemy", False)
    ally = _process_team(a_bx, a_by, a_bw, a_bh, a_n, a_row_h, "ally", True)

    return ally, enemy


def print_team_table(team_name: str, players: list[dict]) -> None:
    """Pretty-print a team's player data."""
    print(f"\n{team_name} ({len(players)} players)")
    print(f"{'Name':<20s} {'E':>5s} {'A':>5s} {'D':>5s} {'DMG':>8s} {'H':>8s} {'MIT':>8s}")
    print("-" * 62)
    for p in players:
        print(f"{p['name']:<20s} {p['E']:>5s} {p['A']:>5s} {p['D']:>5s} "
              f"{p['DMG']:>8s} {p['H']:>8s} {p['MIT']:>8s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OW2 Scoreboard OCR — Stage 1")
    parser.add_argument("image", type=Path, help="Tab screenshot to analyze")
    parser.add_argument("--debug", type=Path, default=None,
                        help="Directory to save debug images")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Ensure tesseract is in PATH
    tess_dir = r"C:\Program Files\Tesseract-OCR"
    if tess_dir not in os.environ["PATH"]:
        os.environ["PATH"] = tess_dir + os.pathsep + os.environ["PATH"]

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Failed to read: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        args.debug.mkdir(parents=True, exist_ok=True)

    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    ally, enemy = process_scoreboard(img, debug_dir=args.debug)
    print_team_table("ALLY TEAM", ally)
    print_team_table("ENEMY TEAM", enemy)


if __name__ == "__main__":
    main()
