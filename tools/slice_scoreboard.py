"""OW2 Scoreboard — detect ally/enemy scoreboards and slice into player rows.

Usage:
    uv run python tools/slice_scoreboard.py screenshot.png [--debug DIR]

Pipeline:
    1. Color mask for ally blue / enemy red backgrounds
    2. Largest contour = scoreboard bounding box
    3. Count rows via white text density peaks in right half
    4. Equal-slice into N rows
"""

import argparse
import sys
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


def extract_rows(img_bgr: np.ndarray, mask: np.ndarray,
                 ) -> tuple[tuple[int, int, int, int], int, int]:
    """Find scoreboard rect and count rows.

    Returns ((bx, by, bw, bh), n_rows, row_h).
    """
    bx, by, bw, bh = find_scoreboard_rect(mask)
    n_rows = count_player_rows(img_bgr, bx, by, bw, bh)
    row_h = bh // n_rows if n_rows > 0 else bh
    return (bx, by, bw, bh), n_rows, row_h


def slice_rows(img_bgr: np.ndarray, bx: int, by: int, bw: int, bh: int,
               n_rows: int, row_h: int) -> list[np.ndarray]:
    """Slice the scoreboard bounding box into individual player row images."""
    rows = []
    for i in range(n_rows):
        y1 = i * row_h
        y2 = (i + 1) * row_h if i < n_rows - 1 else bh
        rows.append(img_bgr[by + y1:by + y2, bx:bx + bw])
    return rows


_PARTY_CHECK_WIDTH = 150  # pixels to scan right of the scoreboard


def detect_party_members(img_bgr: np.ndarray, bx: int, by: int, bw: int,
                         bh: int, n_rows: int, row_h: int,
                         green_threshold: float = 0.10) -> list[bool]:
    """Check which rows have a green party indicator to the right of the scoreboard.

    Scans a strip to the right of the scoreboard bounding box. Only checks
    the middle 50% of each row's height to avoid bleed from adjacent rows.

    Returns a list of bools, one per row (True = in party).
    """
    img_h, img_w = img_bgr.shape[:2]
    right_x = min(bx + bw + _PARTY_CHECK_WIDTH, img_w)

    results = []
    for i in range(n_rows):
        y1 = i * row_h
        y2 = (i + 1) * row_h if i < n_rows - 1 else bh
        mid_h = y2 - y1
        trim = mid_h // 4
        strip = img_bgr[by + y1 + trim:by + y2 - trim, bx + bw:right_x]
        if strip.size == 0:
            results.append(False)
            continue

        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        green = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) &
                 (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50))
        results.append(float(np.mean(green)) >= green_threshold)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OW2 Scoreboard row slicer")
    parser.add_argument("image", type=Path, help="Tab screenshot to analyze")
    parser.add_argument("--debug", type=Path, default=None,
                        help="Directory to save debug images (masks + rows)")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Failed to read: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        args.debug.mkdir(parents=True, exist_ok=True)

    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    for team in ("ally", "enemy"):
        mask = make_color_mask(img, team)
        if args.debug:
            cv2.imwrite(str(args.debug / f"{team}_mask.png"), mask)

        try:
            (bx, by, bw, bh), n_rows, row_h = extract_rows(img, mask)
        except ValueError:
            print(f"{team.upper()}: no scoreboard found")
            continue

        party = detect_party_members(img, bx, by, bw, bh, n_rows, row_h)
        party_str = ", ".join(str(i + 1) for i, p in enumerate(party) if p)

        print(f"{team.upper()}: {n_rows} rows, rect=({bx},{by},{bw},{bh}), row_h={row_h}")
        if party_str:
            print(f"  party slots: {party_str}")

        rows = slice_rows(img, bx, by, bw, bh, n_rows, row_h)
        if args.debug:
            for i, row in enumerate(rows):
                cv2.imwrite(str(args.debug / f"{team}_row_{i:02d}.png"), row)


if __name__ == "__main__":
    main()
