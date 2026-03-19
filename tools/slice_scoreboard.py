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


def find_perk_circles(row_bgr: np.ndarray) -> list[tuple[int, int, int]]:
    """Detect perk icons — white circles with black icons, >50% row height."""
    row_h = row_bgr.shape[0]
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2RGB)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=100, param2=40, minRadius=25, maxRadius=60,
    )
    perks = []
    if circles is not None:
        for x, y, rad in np.round(circles[0]).astype(int):
            if (rad * 2) / row_h < 0.5:
                continue
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), rad, 255, -1)
            pixels = rgb[mask > 0]
            true_black = np.mean((pixels < 60).all(axis=1))
            true_white = np.mean((pixels > 180).all(axis=1))
            if (true_black + true_white) > 0.7 and true_white > 0.3:
                perks.append((x, y, rad))
    return perks


def find_text_groups(row_bgr: np.ndarray,
                     perks: list[tuple[int, int, int]],
                     ) -> list[tuple[int, int, int, int]]:
    """Find white text bounding boxes, merge nearby ones, filter out perks.

    Returns list of (x, y, w, h) sorted left to right.
    """
    row_h, row_w = row_bgr.shape[:2]
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    white = (gray > 140).astype(np.uint8) * 255

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

    # Filter groups that contain a perk circle center
    def contains_perk(gx, gy, gw, gh):
        return any(gx <= px <= gx + gw and gy <= py <= gy + gh
                   for px, py, _ in perks)

    return [tuple(g) for g in groups if not contains_perk(*g)]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_crop(crop_bgr: np.ndarray, lang: str = "eng",
             tessdata: str = r"C:\Program Files\Tesseract-OCR\tessdata") -> str:
    """Run Tesseract PSM 7 (single line) on a BGR image crop."""
    tmp = Path(tempfile.mktemp(suffix=".png"))
    cv2.imwrite(str(tmp), crop_bgr)
    try:
        result = subprocess.run(
            ["tesseract", str(tmp), "stdout", "-l", lang, "--psm", "7"],
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


def process_team(img_bgr: np.ndarray, team: str,
                 debug_dir: Path | None = None) -> list[dict]:
    """Extract all player data from one team's scoreboard.

    Returns list of dicts: {name, E, A, D, DMG, H, MIT}.
    """
    is_ally = team == "ally"
    team_colors = ALLY_BG_COLORS if is_ally else ENEMY_BG_COLORS

    mask = make_color_mask(img_bgr, team)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{team}_mask.png"), mask)

    bx, by, bw, bh = find_scoreboard_rect(mask)
    n_rows = count_player_rows(img_bgr, bx, by, bw, bh)
    row_h = bh // n_rows

    # Ally rows have an ult charge circle as the first group after portrait
    skip_groups = 1 if is_ally else 0
    stat_labels = ["E", "A", "D", "DMG", "H", "MIT"]

    players = []
    for i in range(n_rows):
        y1 = i * row_h
        y2 = (i + 1) * row_h if i < n_rows - 1 else bh
        row_img = img_bgr[by + y1:by + y2, bx:bx + bw]

        # Trim portrait
        cut_x = find_portrait_end(row_img, team_colors)
        trimmed = row_img[:, cut_x:]

        # Detect perks and text groups
        perks = find_perk_circles(trimmed)
        groups = find_text_groups(trimmed, perks)

        # Skip ult charge group for ally
        data_groups = groups[skip_groups:]

        # OCR each group
        values = []
        for gx, gy, gw, gh in data_groups:
            crop = trimmed[gy:gy + gh, gx:gx + gw]
            text = ocr_crop(crop)
            values.append(text)

        # Build player dict
        player = {"name": values[0] if values else ""}
        for j, label in enumerate(stat_labels):
            player[label] = values[j + 1] if j + 1 < len(values) else ""
        players.append(player)

        if debug_dir:
            cv2.imwrite(str(debug_dir / f"{team}_row_{i:02d}.png"), row_img)
            cv2.imwrite(str(debug_dir / f"{team}_row_{i:02d}_trimmed.png"), trimmed)

    return players


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

    ally = process_team(img, "ally", debug_dir=args.debug)
    print_team_table("ALLY TEAM", ally)

    enemy = process_team(img, "enemy", debug_dir=args.debug)
    print_team_table("ENEMY TEAM", enemy)


if __name__ == "__main__":
    main()
