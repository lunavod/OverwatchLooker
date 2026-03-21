"""Read hero panel stats from a tab screenshot. Prints label: value pairs."""

import sys
import cv2
import numpy as np
from pathlib import Path
from paddlex import create_model

# Models
labels_model = create_model("PP-OCRv5_server_rec", model_dir="training_data/panel_labels/inference")
values_model = create_model("PP-OCRv5_server_rec", model_dir="training_data/panel_values_v3/inference")


# ---------------------------------------------------------------------------
# Panel detection (color-run based)
# ---------------------------------------------------------------------------

def detect_panel(img):
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
                if cur > best: best = cur
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
                if cur > best: best = cur
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

def find_text_rows(gray):
    h, w = gray.shape
    bright_pct = (gray > 80).sum(axis=1).astype(float) / w * 100
    text_rows = bright_pct > 1.0
    blocks = []
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
    merged = []
    for s, e in blocks:
        if merged and s - merged[-1][1] < 4:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= 6]


def classify_rows(gray, rows):
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
# Crop helpers
# ---------------------------------------------------------------------------

def crop_to_text(img_bgr, threshold=120, pad=10):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray.max(axis=0) > threshold)[0]
    rows = np.where(gray.max(axis=1) > threshold)[0]
    if len(cols) == 0 or len(rows) == 0:
        return img_bgr
    return img_bgr[
        max(0, rows[0] - pad):rows[-1] + pad,
        max(0, cols[0] - pad):cols[-1] + pad,
    ]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_value(panel, ys, ye):
    """OCR a value row with vertical padding + tight crop."""
    h = panel.shape[0]
    ys_pad = max(0, ys - 20)
    ye_pad = min(h, ye + 20)
    region = panel[ys_pad:ye_pad, :]
    cropped = crop_to_text(region)
    results = list(values_model.predict(cropped))
    if results:
        return results[0]["rec_text"].strip(), results[0]["rec_score"]
    return "", 0.0


def ocr_label(panel, ys, ye):
    """OCR a label row (full width, model handles it)."""
    region = panel[ys:ye, :]
    results = list(labels_model.predict(region))
    if results:
        return results[0]["rec_text"].strip(), results[0]["rec_score"]
    return "", 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_featured_stat(panel):
    """Find the featured stat box (darker rounded rect in top-right)."""
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


def ocr_featured_stat(panel, box):
    """OCR the featured stat box — value on top, label on bottom."""
    bx, by, bw, bh = box
    featured = panel[by:by + bh, bx:bx + bw]
    h = featured.shape[0]

    # Top ~60% is value, bottom ~40% is label
    split_y = int(h * 0.6)
    val_region = crop_to_text(featured[:split_y, :])
    lbl_region = featured[split_y:, :]

    vr = list(values_model.predict(val_region))
    val = vr[0]["rec_text"].strip() if vr else ""
    vs = vr[0]["rec_score"] if vr else 0

    lr = list(labels_model.predict(lbl_region))
    lbl = lr[0]["rec_text"].strip() if lr else ""
    ls = lr[0]["rec_score"] if lr else 0

    return lbl, val, ls, vs


def read_hero_panel(img):
    """Extract featured stat + stat label:value pairs from a tab screenshot."""
    rect = detect_panel(img)
    if rect is None:
        print("No hero panel found.")
        return None, []

    px, py, pw, ph = rect
    panel = img[py:py + ph, px:px + pw]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    # Featured stat
    featured = None
    box = detect_featured_stat(panel)
    if box:
        featured = ocr_featured_stat(panel, box)

    # Regular stats
    rows = find_text_rows(gray)
    classified = classify_rows(gray, rows)

    value_rows = [(s, e) for s, e, t in classified if t == "value"][1:]  # skip hero name
    label_rows = [(s, e) for s, e, t in classified if t == "label"]

    n = min(len(value_rows), len(label_rows))
    stats = []
    for i in range(n):
        val_text, val_score = ocr_value(panel, *value_rows[i])
        lbl_text, lbl_score = ocr_label(panel, *label_rows[i])
        stats.append((lbl_text, val_text, lbl_score, val_score))

    return featured, stats


if __name__ == "__main__":
    images = sys.argv[1:] if len(sys.argv) > 1 else [
        "test_two_perks.png",
        "rein_one_perk.png",
        "full_of_stuff.png",
    ]

    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read {path}")
            continue

        print(f"\n{'='*50}")
        print(f"  {Path(path).name}")
        print(f"{'='*50}")

        featured, stats = read_hero_panel(img)
        if featured:
            lbl, val, ls, vs = featured
            print(f"  * {lbl}: {val}  (featured)")
        for label, value, lscore, vscore in stats:
            print(f"  {label}: {value}")
