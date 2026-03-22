"""Detect Control mode score (round wins) from an Overwatch 2 gameplay screenshot.

Looks for filled blue/red score circles in the top-center HUD. Control maps
show 2 small circles per team — filled = round won, empty ring = not yet won.

Usage:
    uv run python tools/detect_control_score.py screenshot.png [--debug DIR]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ROI: top-center of screen where score HUD lives
_ROI_Y = (45, 105)       # pixel range at 1080p
_ROI_X = (0.38, 0.62)    # fraction of width

# Color thresholds (HSV)
# Blue filled circles: cyan-blue, high saturation + value
_BLUE_H = (85, 115)
_BLUE_SV_MIN = 150

# Red/pink filled circles: red/magenta hue, high saturation + value
_RED_H_LOW = 8       # 0..8
_RED_H_HIGH = 155    # 155..180
_RED_SV_MIN = 150

# Score circle regions: circles are outside the percentage boxes
_BLUE_X_FRAC = 0.30   # left 30% of ROI
_RED_X_FRAC = 0.70     # right 30% of ROI

# Circle contour filters (at 1080p)
_MIN_AREA = 400
_MAX_AREA = 1200
_MIN_CIRCULARITY = 0.70
_ASPECT_RANGE = (0.7, 1.4)
_SIZE_RANGE = (20, 45)


def _make_masks(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create blue and red binary masks from the score ROI."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blue = ((hsv[:, :, 0] > _BLUE_H[0]) & (hsv[:, :, 0] < _BLUE_H[1]) &
            (hsv[:, :, 1] > _BLUE_SV_MIN) & (hsv[:, :, 2] > _BLUE_SV_MIN)
            ).astype(np.uint8) * 255

    red = (((hsv[:, :, 0] > _RED_H_HIGH) | (hsv[:, :, 0] < _RED_H_LOW)) &
           (hsv[:, :, 1] > _RED_SV_MIN) & (hsv[:, :, 2] > _RED_SV_MIN)
           ).astype(np.uint8) * 255

    return blue, red


def _count_filled_circles(mask: np.ndarray) -> int:
    """Count filled score circles in a binary mask via contour analysis."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        _, _, cw, ch = cv2.boundingRect(c)

        if (circularity > _MIN_CIRCULARITY
                and _MIN_AREA < area < _MAX_AREA
                and _ASPECT_RANGE[0] < cw / ch < _ASPECT_RANGE[1]
                and _SIZE_RANGE[0] < cw < _SIZE_RANGE[1]):
            count += 1

    return count


def detect_control_score(img: np.ndarray) -> tuple[int, int] | None:
    """Detect blue and red round scores from a Control mode screenshot.

    Returns (blue_score, red_score) or None if no score HUD is detected.
    Scores are 0, 1, or 2.
    """
    h, w = img.shape[:2]

    # Scale ROI Y coordinates for non-1080p
    scale = h / 1080.0
    y1 = int(_ROI_Y[0] * scale)
    y2 = int(_ROI_Y[1] * scale)
    x1 = int(w * _ROI_X[0])
    x2 = int(w * _ROI_X[1])

    roi = img[y1:y2, x1:x2]
    rw = roi.shape[1]

    blue_mask, red_mask = _make_masks(roi)

    blue_region = blue_mask[:, :int(rw * _BLUE_X_FRAC)]
    red_region = red_mask[:, int(rw * _RED_X_FRAC):]

    # Scale area/size thresholds for non-1080p
    area_scale = scale * scale
    min_area = int(_MIN_AREA * area_scale)
    max_area = int(_MAX_AREA * area_scale)
    min_size = int(_SIZE_RANGE[0] * scale)
    max_size = int(_SIZE_RANGE[1] * scale)

    def _count(mask: np.ndarray) -> int:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        for c in contours:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if perim == 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            _, _, cw, ch = cv2.boundingRect(c)
            if (circularity > _MIN_CIRCULARITY
                    and min_area < area < max_area
                    and _ASPECT_RANGE[0] < cw / ch < _ASPECT_RANGE[1]
                    and min_size < cw < max_size):
                n += 1
        return n

    blue_score = _count(blue_region)
    red_score = _count(red_region)

    return (blue_score, red_score)


def main():
    parser = argparse.ArgumentParser(
        description="Detect Control mode score from OW2 screenshot")
    parser.add_argument("image", type=Path, help="Screenshot to analyze")
    parser.add_argument("--debug", type=Path, default=None,
                        help="Directory to save debug images")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Failed to read: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    if args.debug:
        args.debug.mkdir(parents=True, exist_ok=True)

        h, w = img.shape[:2]
        scale = h / 1080.0
        y1, y2 = int(_ROI_Y[0] * scale), int(_ROI_Y[1] * scale)
        x1, x2 = int(w * _ROI_X[0]), int(w * _ROI_X[1])
        roi = img[y1:y2, x1:x2]
        rw = roi.shape[1]

        cv2.imwrite(str(args.debug / "roi.png"), roi)

        blue_mask, red_mask = _make_masks(roi)
        cv2.imwrite(str(args.debug / "blue_mask.png"), blue_mask)
        cv2.imwrite(str(args.debug / "red_mask.png"), red_mask)

        cv2.imwrite(str(args.debug / "blue_region.png"),
                    blue_mask[:, :int(rw * _BLUE_X_FRAC)])
        cv2.imwrite(str(args.debug / "red_region.png"),
                    red_mask[:, int(rw * _RED_X_FRAC):])

    result = detect_control_score(img)
    if result is None:
        print("No score detected")
    else:
        blue, red = result
        print(f"Score: {blue} - {red} (blue - red)")


if __name__ == "__main__":
    main()
