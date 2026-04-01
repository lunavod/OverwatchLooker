"""Detect and OCR hero SR cards from the post-match rank screen.

Uses hero portrait template matching to find cards, then extracts
SR total and SR delta via OCR.

Usage:
    uv run python tools/crop_hero_sr.py <frame.png>
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from paddlex import create_model

_HEROES_DIR = Path(__file__).parent.parent / "overwatchlooker" / "assets" / "heroes"
_VALUES_MODEL_DIR = Path(__file__).parent.parent / "overwatchlooker" / "models" / "panel_values"

# Search region for hero portraits (right side of screen)
_SEARCH_X1 = 1750
_SEARCH_Y1 = 350
_SEARCH_X2 = 1920
_SEARCH_Y2 = 650

# Portrait matching settings
_MATCH_THRESHOLD = 0.8
_SCALES = [0.25, 0.28, 0.31, 0.34, 0.37, 0.40]


def load_portrait_templates() -> dict[str, np.ndarray]:
    """Load hero portraits, crop to middle 40% (remove 30% each edge)."""
    templates = {}
    for path in sorted(_HEROES_DIR.glob("*.png")):
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        h, w = raw.shape[:2]
        margin = int(w * 0.3)
        inner = raw[margin:h - margin, margin:w - margin, :3]  # BGR only
        templates[path.stem] = inner
    return templates


def find_hero_cards(img: np.ndarray,
                    templates: dict[str, np.ndarray]
                    ) -> list[tuple[str, int, int, int, int, float]]:
    """Find hero SR cards by multi-scale template matching.

    Returns list of (hero_name, x, y, w, h, score) sorted by y position.
    """
    roi = img[_SEARCH_Y1:_SEARCH_Y2, _SEARCH_X1:_SEARCH_X2]
    rh, rw = roi.shape[:2]

    matches = []
    for hero, tmpl in templates.items():
        th, tw = tmpl.shape[:2]
        best_val = -1.0
        best_loc = (0, 0)
        best_size = (0, 0)
        for scale in _SCALES:
            sw, sh = int(tw * scale), int(th * scale)
            if sw > rw or sh > rh or sw < 10 or sh < 10:
                continue
            resized = cv2.resize(tmpl, (sw, sh))
            result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(result)
            if mv > best_val:
                best_val = mv
                best_loc = ml
                best_size = (sw, sh)
        if best_val >= _MATCH_THRESHOLD:
            mx, my = best_loc
            sw, sh = best_size
            matches.append((hero, _SEARCH_X1 + mx, _SEARCH_Y1 + my,
                            sw, sh, best_val))

    # Remove overlapping matches (keep best score per y-region)
    matches.sort(key=lambda m: m[5], reverse=True)
    filtered = []
    for m in matches:
        overlap = False
        for f in filtered:
            if abs(m[2] - f[2]) < 50:
                overlap = True
                break
        if not overlap:
            filtered.append(m)

    filtered.sort(key=lambda m: m[2])
    return filtered


def binarize_white(img: np.ndarray, threshold: int = 170) -> np.ndarray | None:
    """Binarize white text, crop to bbox. Returns BGR or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return None
    pad = max(5, int((ys.max() - ys.min()) * 0.3))
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(binary.shape[0], int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(binary.shape[1], int(xs.max()) + pad)
    cropped = binary[y1:y2, x1:x2]
    padded = cv2.copyMakeBorder(cropped, 5, 5, 5, 5,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def binarize_delta(img: np.ndarray, min_h: int = 8) -> tuple[np.ndarray | None, str]:
    """Binarize colored delta text, remove icons/signs. Returns (BGR, sign) or (None, sign).

    Sign is '+' for green, '-' for red.
    """
    r, g, b = img[:, :, 2].astype(int), img[:, :, 1].astype(int), img[:, :, 0].astype(int)
    red = (r > 120) & (r > g * 2) & (r > b * 2)
    green = (g > 120) & (g > r * 2) & (g > b * 2)

    sign = "-" if np.sum(red) > np.sum(green) else "+"

    binary = np.where(red | green, 255, 0).astype(np.uint8)

    # Erase short contours (arrow icons, minus sign) — keep only digits
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        _, _, w2, h2 = cv2.boundingRect(c)
        if h2 < min_h:
            cv2.drawContours(binary, [c], -1, 0, cv2.FILLED)

    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return None, sign
    pad = max(5, int((ys.max() - ys.min()) * 0.3))
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(binary.shape[0], int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(binary.shape[1], int(xs.max()) + pad)
    cropped = binary[y1:y2, x1:x2]
    padded = cv2.copyMakeBorder(cropped, 5, 5, 5, 5,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), sign


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <frame.png>")
        sys.exit(1)

    frame_path = Path(sys.argv[1])
    if not frame_path.exists():
        print(f"File not found: {frame_path}")
        sys.exit(1)

    img = cv2.imread(str(frame_path))
    h, w = img.shape[:2]
    print(f"Frame: {w}x{h}")

    out_dir = frame_path.parent / "debug_hero_sr"
    out_dir.mkdir(exist_ok=True)

    print("Loading models...")
    templates = load_portrait_templates()
    print(f"  {len(templates)} hero portraits")
    values_model = create_model(model_name="PP-OCRv5_server_rec",
                                model_dir=str(_VALUES_MODEL_DIR))

    print("\nMatching portraits...")
    cards = find_hero_cards(img, templates)

    annotated = img.copy()
    out_dir = frame_path.parent / "debug_hero_sr"
    out_dir.mkdir(exist_ok=True)

    for hero, x, y, w, h, score in cards:
        print(f"  {hero}: ({x},{y}) {w}x{h} score={score:.4f}")

        # Draw portrait match box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find left edge of portrait border by scanning for cyan line
        # Portrait border cyan ~rgb(18, 206, 251) = BGR(251, 206, 18)
        cyan_bgr = np.array([251, 206, 18], dtype=float)
        full_size = int(w / 0.4)
        cy = y + h // 2
        card_y1 = cy - full_size // 2 - 5
        card_y2 = cy + full_size // 2 + 25

        # Scan leftward from portrait match to find the cyan border
        scan_row = img[cy - 5:cy + 5, x - 120:x + w]
        scan_f = scan_row.astype(float)
        cyan_dist = np.sqrt(np.sum((scan_f - cyan_bgr) ** 2, axis=2))
        cyan_cols = np.where(np.min(cyan_dist, axis=0) < 50)[0]
        if len(cyan_cols) > 0:
            # Leftmost cyan column relative to the scan start
            border_left = (x - 120) + int(cyan_cols[0])
            card_x2 = border_left - 5
        else:
            card_x2 = x - 30
        card_x1 = card_x2 - 130

        # Detect calibration squares: 5 similar small contours in a vertical column
        content_region = img[max(0, card_y1):card_y2, max(0, card_x1):card_x2]
        gray_content = cv2.cvtColor(content_region, cv2.COLOR_BGR2GRAY)
        # Squares are either white (~200+) or dark (~40-80) on medium bg (~100-150)
        # Use edge detection to find them regardless of fill
        edges = cv2.Canny(gray_content, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find groups of ~5 contours with similar size in a vertical column
        rects = [cv2.boundingRect(c) for c in cnts]
        rects = [(x2, y2, w2, h2) for x2, y2, w2, h2 in rects
                 if 4 < w2 < 25 and 4 < h2 < 25 and abs(w2 - h2) < 8]

        # Group by similar x position (within 5px)
        if rects:
            rects.sort(key=lambda r: r[0])
            columns: list[list[tuple]] = [[rects[0]]]
            for r in rects[1:]:
                if abs(r[0] - columns[-1][0][0]) < 8:
                    columns[-1].append(r)
                else:
                    columns.append([r])

            # Find a column with 3-5 similarly-sized squares
            for col in columns:
                if len(col) < 3:
                    continue
                widths = [r[2] for r in col]
                heights = [r[3] for r in col]
                if max(widths) - min(widths) < 6 and max(heights) - min(heights) < 6:
                    # Found calibration squares — cut content to the left of them
                    sq_x = min(r[0] for r in col) - 3
                    card_x2 = card_x1 + sq_x
                    break

        # Split content into 3 lines by finding horizontal gaps in brightness
        content = img[max(0, card_y1):card_y2, max(0, card_x1):card_x2]
        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        row_max = np.max(gray, axis=1)

        # Find rows with text: bright white OR red/green delta text
        # Red delta: R channel dominant (R > 100, R > G*2, R > B*1.5)
        r, g, b = content[:, :, 2], content[:, :, 1], content[:, :, 0]
        red_delta = (r > 100) & (r > g * 2) & (r > b * 1.5)
        # Green delta: G channel dominant (G > 100, G > R*2, G > B*2)
        green_delta = (g > 100) & (g > r * 2) & (g > b * 2)
        delta_rows = np.sum(red_delta | green_delta, axis=1) > 3

        text_rows = (row_max > 100) | delta_rows
        # Find transitions to split into lines
        lines = []
        in_text = False
        start = 0
        for ry in range(len(text_rows)):
            if text_rows[ry] and not in_text:
                start = ry
                in_text = True
            elif not text_rows[ry] and in_text:
                lines.append((start, ry))
                in_text = False
        if in_text:
            lines.append((start, len(text_rows)))

        line_names = ["label", "sr_total", "sr_delta"]
        line_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255)]

        sr_total = None
        sr_delta = None
        delta_sign = None

        for i, (ly1, ly2) in enumerate(lines[:3]):
            name = line_names[i] if i < len(line_names) else f"line_{i}"
            color = line_colors[i] if i < len(line_colors) else (128, 128, 128)

            abs_y1 = max(0, card_y1) + ly1
            abs_y2 = max(0, card_y1) + ly2
            cv2.rectangle(annotated, (card_x1, abs_y1), (card_x2, abs_y2), color, 1)

            line_crop = content[ly1:ly2, :]
            cv2.imwrite(str(out_dir / f"{hero}_{name}.png"), line_crop)

            if name == "sr_total":
                ocr_img = binarize_white(line_crop)
                if ocr_img is not None:
                    result = list(values_model.predict(ocr_img))
                    sr_total = result[0]["rec_text"]
                    cv2.imwrite(str(out_dir / f"{hero}_sr_total_ocr.png"), ocr_img)

            elif name == "sr_delta":
                # Upscale 3x before binarizing for better small-text OCR
                upscaled = cv2.resize(line_crop, None, fx=3, fy=3,
                                      interpolation=cv2.INTER_CUBIC)
                ocr_img, delta_sign = binarize_delta(upscaled, min_h=20)
                if ocr_img is not None:
                    result = list(values_model.predict(ocr_img))
                    sr_delta = result[0]["rec_text"]
                    cv2.imwrite(str(out_dir / f"{hero}_sr_delta_ocr.png"), ocr_img)

        print(f"    SR: {sr_total}")
        if sr_delta and delta_sign:
            print(f"    Delta: {delta_sign}{sr_delta}")
        elif delta_sign:
            print(f"    Delta: {delta_sign}? (no text)")

        cv2.rectangle(annotated, (card_x1, card_y1), (card_x2, card_y2),
                      (0, 255, 255), 1)
        cv2.putText(annotated, hero, (card_x1, card_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save full content crop
        crop = img[max(0, card_y1):card_y2, max(0, card_x1):card_x2]
        cv2.imwrite(str(out_dir / f"{hero}_content.png"), crop)

    if not cards:
        print("  No hero cards found")

    ann_path = out_dir / "annotated.png"
    cv2.imwrite(str(ann_path), annotated)
    print(f"\nannotated -> {ann_path}")


if __name__ == "__main__":
    main()
