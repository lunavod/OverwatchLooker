"""Crop regions of interest from the post-match rank screen and OCR them.

Also binarizes and crops to text bounds for OCR-ready output.
All pixel coordinates are defined at 1920x1080 and scaled to actual resolution.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from paddlex import create_model

# Reference resolution — all region coordinates are defined at this size.
_REF_W, _REF_H = 1920, 1080

# Regions defined as (x1, y1, x2, y2) in pixels at 1920x1080.
REGIONS = {
    # "GOLD 3" text below the rank emblem
    "rank_division": (720, 670, 1200, 730),

    # "RANK PROGRESS: 24%" line above the progress bar
    "rank_progress": (480, 790, 750, 825),

    # The progress bar itself (green/red segment + delta text)
    "progress_bar": (480, 825, 1440, 870),

    # Modifier tags below the progress bar (VICTORY, UPHILL BATTLE, etc.)
    "modifiers": (480, 878, 1440, 923),
}

_TEXT_THRESH = 140  # brightness threshold for white text

_MODELS_DIR = Path(__file__).parent.parent / "overwatchlooker" / "models"
_RANK_MODEL_DIR = Path(__file__).parent.parent / "training_data" / "rank_division" / "inference"
_VALUES_MODEL_DIR = _MODELS_DIR / "panel_values"
_DELTA_MODEL_DIR = Path(__file__).parent.parent / "training_data" / "progress_delta" / "inference"
_MODIFIERS_MODEL_DIR = Path(__file__).parent.parent / "training_data" / "modifiers" / "inference"

# Colored number targets in BGR (OpenCV order)
_TEAL_BGR = np.array([237, 253, 103])   # positive progress
_ORANGE_BGR = np.array([35, 95, 212])   # negative progress
_COLOR_TOLERANCE = 80  # max Euclidean distance from target color
_CHEVRON_UPSCALE = 4   # upscale factor for chevron shape analysis

# Template chevron contours for shape matching (drawn at high res, scale-invariant)
def _make_chevron_contour(direction: str) -> np.ndarray:
    sz = 200
    img = np.zeros((sz, sz), dtype=np.uint8)
    if direction == "right":
        pts = np.array([[sz//4, sz//6], [3*sz//4, sz//2], [sz//4, 5*sz//6]], np.int32)
    else:
        pts = np.array([[3*sz//4, sz//6], [sz//4, sz//2], [3*sz//4, 5*sz//6]], np.int32)
    cv2.polylines(img, [pts], False, 255, 8)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0]

_CHEVRON_RIGHT = _make_chevron_contour("right")
_CHEVRON_LEFT = _make_chevron_contour("left")
_CHEVRON_MATCH_THRESH = 0.5


def _scale_region(name: str, sx: float, sy: float) -> tuple[int, int, int, int]:
    """Scale a reference region from 1080p to the actual resolution."""
    x1, y1, x2, y2 = REGIONS[name]
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def binarize_bbox(gray: np.ndarray, threshold: int = _TEXT_THRESH,
                  pad_frac: float = 0.3, min_pad: int = 10) -> np.ndarray:
    """Binarize, crop to text bbox, pad with black. Returns BGR for model."""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pad = max(min_pad, int((ys.max() - ys.min()) * pad_frac))
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(binary.shape[0], int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(binary.shape[1], int(xs.max()) + pad)
    cropped = binary[y1:y2, x1:x2]
    padded = cv2.copyMakeBorder(cropped, 5, 5, 5, 5,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def isolate_colored_text(crop_bgr: np.ndarray,
                         pad_frac: float = 0.3, min_pad: int = 10) -> np.ndarray:
    """Mask out everything except teal/orange colored text, crop to bbox."""
    img_f = crop_bgr.astype(float)
    dist_teal = np.sqrt(np.sum((img_f - _TEAL_BGR.astype(float)) ** 2, axis=2))
    dist_orange = np.sqrt(np.sum((img_f - _ORANGE_BGR.astype(float)) ** 2, axis=2))
    mask = (dist_teal < _COLOR_TOLERANCE) | (dist_orange < _COLOR_TOLERANCE)
    binary = np.where(mask, 255, 0).astype(np.uint8)

    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    pad = max(min_pad, int((ys.max() - ys.min()) * pad_frac))
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(binary.shape[0], int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(binary.shape[1], int(xs.max()) + pad)
    cropped = binary[y1:y2, x1:x2]
    padded = cv2.copyMakeBorder(cropped, 5, 5, 5, 5,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def _is_chevron(contour: np.ndarray, scale: float = 1.0) -> bool:
    """Detect if an upscaled contour is a chevron (> or <) by geometric shape.

    Chevrons have: h/w >= 1.4, simplify to 3-5 polygon vertices,
    top/bottom at similar x, tip offset horizontally near vertical center.
    """
    area = cv2.contourArea(contour)
    if area < 800 * scale ** 2:  # too small at 4x — % dots, noise
        return False
    x, y, w, h = cv2.boundingRect(contour)
    if w < 8 * scale or h < 8 * scale:
        return False
    if h < w * 1.4:
        return False

    peri = cv2.arcLength(contour, True)
    for eps in [0.08, 0.06, 0.04]:
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if 3 <= len(approx) <= 5:
            break
    else:
        return False
    if len(approx) < 3 or len(approx) > 5:
        return False

    pts = approx.squeeze().tolist()
    top = min(pts, key=lambda p: p[1])
    bot = max(pts, key=lambda p: p[1])
    left = min(pts, key=lambda p: p[0])
    right = max(pts, key=lambda p: p[0])
    mid_y = (top[1] + bot[1]) / 2

    if abs(top[0] - bot[0]) > w * 0.4:
        return False
    if abs(right[1] - mid_y) < h * 0.35 and (right[0] - min(top[0], bot[0])) > w * 0.4:
        return True
    if abs(left[1] - mid_y) < h * 0.35 and (max(top[0], bot[0]) - left[0]) > w * 0.4:
        return True
    return False


def _remove_chevrons(mask: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Remove chevron shapes from a binary mask using upscaled contour analysis."""
    upscaled = cv2.resize(mask, None, fx=_CHEVRON_UPSCALE, fy=_CHEVRON_UPSCALE,
                          interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(upscaled, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 20 * scale ** 2:
            continue
        if _is_chevron(c, scale):
            cv2.drawContours(upscaled, [c], -1, 0, cv2.FILLED)

    return cv2.resize(upscaled, (mask.shape[1], mask.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def detect_progress_sign(crop_bgr: np.ndarray) -> str:
    """Detect whether progress is positive or negative from dominant color."""
    img_f = crop_bgr.astype(float)
    teal_pixels = np.sum(np.sqrt(np.sum((img_f - _TEAL_BGR.astype(float)) ** 2, axis=2)) < _COLOR_TOLERANCE)
    orange_pixels = np.sum(np.sqrt(np.sum((img_f - _ORANGE_BGR.astype(float)) ** 2, axis=2)) < _COLOR_TOLERANCE)
    if orange_pixels > teal_pixels:
        return "-"
    return "+"


def ocr_rank_progress(img: np.ndarray, model) -> tuple[str, float, str]:
    """Extract rank progress percentage from colored numbers.

    Returns (text, score, sign) where sign is '+' or '-'.
    """
    h, w = img.shape[:2]
    sx, sy = w / _REF_W, h / _REF_H
    x1, y1, x2, y2 = _scale_region("rank_progress", sx, sy)
    crop = img[y1:y2, x1:x2]
    sign = detect_progress_sign(crop)
    ocr_img = isolate_colored_text(crop)
    result = list(model.predict(ocr_img))
    return result[0]["rec_text"], result[0]["rec_score"], sign


def extract_progress_bar_delta(crop_bgr: np.ndarray,
                               scale: float = 1.0,
                               pad_frac: float = 0.3, min_pad: int = 10
                               ) -> tuple[np.ndarray | None, str | None]:
    """Extract white text inside the green/red delta segment of the progress bar.

    Returns (ocr_ready_bgr, sign) or (None, None) if demotion protection.
    """
    # Target colors in BGR
    green_bgr_1 = np.array([36, 234, 76], dtype=float)  # compressed/static
    green_bgr_2 = np.array([0, 252, 3], dtype=float)   # raw video
    red_bgr = np.array([95, 47, 181], dtype=float)
    tolerance = 80

    img_f = crop_bgr.astype(float)
    green_mask = ((np.sqrt(np.sum((img_f - green_bgr_1) ** 2, axis=2)) < tolerance) |
                  (np.sqrt(np.sum((img_f - green_bgr_2) ** 2, axis=2)) < tolerance))
    red_mask = np.sqrt(np.sum((img_f - red_bgr) ** 2, axis=2)) < tolerance

    green_count = int(np.sum(green_mask))
    red_count = int(np.sum(red_mask))

    pixel_thresh = int(50 * scale ** 2)
    if green_count < pixel_thresh and red_count < pixel_thresh:
        return None, None  # demotion protection

    sign = "+" if green_count > red_count else "-"

    # Find bounding box of the colored segment
    color_mask = (green_mask | red_mask)
    ys, xs = np.where(color_mask)
    cx1, cx2 = int(xs.min()), int(xs.max())
    cy1, cy2 = int(ys.min()), int(ys.max())

    # Clean the color mask: remove thin connections from anti-aliasing
    color_mask_u8 = color_mask.astype(np.uint8) * 255
    ks = max(3, int(round(3 * scale)))
    if ks % 2 == 0:
        ks += 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    color_mask_u8 = cv2.morphologyEx(color_mask_u8, cv2.MORPH_OPEN, open_kernel)
    contours, _ = cv2.findContours(color_mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, sign

    # Fill the largest contour to get the full segment including text holes
    largest = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(filled_mask, [largest], -1, 255, cv2.FILLED)

    # White pixels inside the filled contour, excluding the colored pixels themselves
    # Use a high threshold — text is near-pure white, avoids anti-aliasing bleed
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    not_colored = cv2.bitwise_not(color_mask_u8)
    text_only = cv2.bitwise_and(white_mask, cv2.bitwise_and(filled_mask, not_colored))

    # Remove chevron shapes
    text_only = _remove_chevrons(text_only, scale)

    # Strip the +/- sign (leftmost contour) — sign comes from color,
    # and unknown chars in the values model corrupt the CTC decoder
    strip_cnts, _ = cv2.findContours(text_only, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if strip_cnts and len(strip_cnts) > 1:
        by_x = sorted(strip_cnts, key=lambda c: cv2.boundingRect(c)[0])
        first = cv2.boundingRect(by_x[0])
        second = cv2.boundingRect(by_x[1])
        # Only strip if there's a gap (sign is separate from digits)
        if second[0] - (first[0] + first[2]) >= max(1, int(scale)):
            cv2.drawContours(text_only, [by_x[0]], -1, 0, cv2.FILLED)

    ys, xs = np.where(text_only > 0)
    if len(ys) == 0:
        return None, sign
    bx1, bx2 = int(xs.min()), int(xs.max())
    by1, by2 = int(ys.min()), int(ys.max())

    pad = max(min_pad, int((by2 - by1) * pad_frac))
    y1 = max(0, by1 - pad)
    y2 = min(text_only.shape[0], by2 + pad)
    x1 = max(0, bx1 - pad)
    x2 = min(text_only.shape[1], bx2 + pad)
    cropped = text_only[y1:y2, x1:x2]
    padded = cv2.copyMakeBorder(cropped, 5, 5, 5, 5,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), sign


def ocr_progress_bar(img: np.ndarray, model) -> tuple[str | None, float, str | None]:
    """Extract delta percentage from the progress bar.

    Returns (text, score, sign) or (None, 0, None) for demotion protection.
    Sign is detected from bar color (green=+, red=-).
    """
    h, w = img.shape[:2]
    sx, sy = w / _REF_W, h / _REF_H
    scale = (sx + sy) / 2
    x1, y1, x2, y2 = _scale_region("progress_bar", sx, sy)
    crop = img[y1:y2, x1:x2]
    ocr_img, color_sign = extract_progress_bar_delta(crop, scale)
    if ocr_img is None and color_sign is None:
        return None, 0.0, None
    if ocr_img is None:
        return None, 0.0, color_sign
    result = list(model.predict(ocr_img))
    raw = result[0]["rec_text"]
    score = result[0]["rec_score"]
    # Clean repeated % signs (model quirk on small crops)
    if "%" in raw:
        text = raw.split("%")[0] + "%"
    else:
        text = raw
    return text, score, color_sign


def ocr_modifiers(img: np.ndarray, model) -> list[tuple[str, float]]:
    """Extract modifier labels from the modifiers region.

    Returns list of (text, score) tuples, one per modifier.
    """
    fh, fw = img.shape[:2]
    sx, sy = fw / _REF_W, fh / _REF_H
    x1, y1, x2, y2 = _scale_region("modifiers", sx, sy)
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Remove colored icons: fill contours of high-sat + high-value pixels
    icon_mask = ((hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 130)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(icon_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    icon_filled = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(icon_filled, contours, -1, 255, cv2.FILLED)

    # White text only, excluding icon interiors
    text_mask = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 160)).astype(np.uint8) * 255
    text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(icon_filled))

    # Split into groups by large x-gaps
    cnts, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    rects = sorted([cv2.boundingRect(c) for c in cnts], key=lambda r: r[0])
    groups: list[list[tuple[int, int, int, int]]] = [[rects[0]]]
    for r in rects[1:]:
        prev = groups[-1][-1]
        gap = r[0] - (prev[0] + prev[2])
        avg_h = float(np.mean([rr[3] for rr in groups[-1]]))
        if gap > avg_h * 1.5:
            groups.append([])
        groups[-1].append(r)

    # OCR each group
    pad = max(5, int(5 * sx))
    results = []
    for group in groups:
        gx1 = min(r[0] for r in group)
        gy1 = min(r[1] for r in group)
        gx2 = max(r[0] + r[2] for r in group)
        gy2 = max(r[1] + r[3] for r in group)
        cy1 = max(0, gy1 - pad)
        cy2 = min(text_mask.shape[0], gy2 + pad)
        cx1 = max(0, gx1 - pad)
        cx2 = min(text_mask.shape[1], gx2 + pad)
        cropped = text_mask[cy1:cy2, cx1:cx2]
        padded = cv2.copyMakeBorder(cropped, 5, 5, 5, 5,
                                    cv2.BORDER_CONSTANT, value=0)
        ocr_img = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
        result = list(model.predict(ocr_img))
        results.append((result[0]["rec_text"], result[0]["rec_score"]))

    return results


def ocr_rank_division(img: np.ndarray, model) -> tuple[str, float]:
    """Extract rank + division text from a full frame. Returns (text, score)."""
    h, w = img.shape[:2]
    sx, sy = w / _REF_W, h / _REF_H
    x1, y1, x2, y2 = _scale_region("rank_division", sx, sy)
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ocr_img = binarize_bbox(gray)
    result = list(model.predict(ocr_img))
    return result[0]["rec_text"], result[0]["rec_score"]


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
    sx, sy = w / _REF_W, h / _REF_H
    scale = (sx + sy) / 2
    print(f"Frame: {w}x{h} (scale: {sx:.2f}x)")

    out_dir = frame_path.parent / "debug_rank_crops"
    out_dir.mkdir(exist_ok=True)

    for name, (rx1, ry1, rx2, ry2) in REGIONS.items():
        x1, y1 = int(rx1 * sx), int(ry1 * sy)
        x2, y2 = int(rx2 * sx), int(ry2 * sy)
        crop = img[y1:y2, x1:x2]
        out_path = out_dir / f"{name}.png"
        cv2.imwrite(str(out_path), crop)
        print(f"  {name}: ({x1},{y1})-({x2},{y2}) -> {out_path}")

    # Binarize + crop to text for rank_division
    x1, y1, x2, y2 = _scale_region("rank_division", sx, sy)
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ocr_ready = binarize_bbox(gray)
    ocr_path = out_dir / "rank_division_ocr.png"
    cv2.imwrite(str(ocr_path), ocr_ready)
    print(f"  rank_division_ocr -> {ocr_path} ({ocr_ready.shape[1]}x{ocr_ready.shape[0]})")

    # Isolate colored text for rank_progress
    x1, y1, x2, y2 = _scale_region("rank_progress", sx, sy)
    rp_crop = img[y1:y2, x1:x2]
    rp_ocr = isolate_colored_text(rp_crop)
    rp_ocr_path = out_dir / "rank_progress_ocr.png"
    cv2.imwrite(str(rp_ocr_path), rp_ocr)
    print(f"  rank_progress_ocr -> {rp_ocr_path} ({rp_ocr.shape[1]}x{rp_ocr.shape[0]})")

    # OCR
    print("\nLoading models...")
    rank_model = create_model(model_name="PP-OCRv5_server_rec",
                              model_dir=str(_RANK_MODEL_DIR))
    values_model = create_model(model_name="PP-OCRv5_server_rec",
                                model_dir=str(_VALUES_MODEL_DIR))
    modifiers_model = create_model(model_name="PP-OCRv5_server_rec",
                                   model_dir=str(_MODIFIERS_MODEL_DIR))
    text, score = ocr_rank_division(img, rank_model)
    print(f"  Rank: {text} ({score:.4f})")

    text, score, sign = ocr_rank_progress(img, values_model)
    print(f"  Progress: {sign}{text} ({score:.4f})")

    # Progress bar delta
    x1, y1, x2, y2 = _scale_region("progress_bar", sx, sy)
    pb_crop = img[y1:y2, x1:x2]
    pb_ocr, pb_sign = extract_progress_bar_delta(pb_crop, scale)
    if pb_ocr is not None:
        pb_ocr_path = out_dir / "progress_bar_ocr.png"
        cv2.imwrite(str(pb_ocr_path), pb_ocr)
        print(f"  progress_bar_ocr -> {pb_ocr_path} ({pb_ocr.shape[1]}x{pb_ocr.shape[0]})")

    delta_text, delta_score, delta_sign = ocr_progress_bar(img, values_model)
    if delta_text is None and delta_sign is None:
        print(f"  Delta: DEMOTION PROTECTION")
    elif delta_text is None:
        print(f"  Delta: {delta_sign}? (no text found)")
    else:
        print(f"  Delta: {delta_sign}{delta_text} ({delta_score:.4f})")

    # Modifiers
    mods = ocr_modifiers(img, modifiers_model)
    if mods:
        print(f"  Modifiers: {', '.join(m[0] for m in mods)}")
        for m_text, m_score in mods:
            print(f"    - {m_text} ({m_score:.4f})")
    else:
        print(f"  Modifiers: none")

    # Also save annotated full frame with rectangles
    annotated = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
    for i, (name, (rx1, ry1, rx2, ry2)) in enumerate(REGIONS.items()):
        x1, y1 = int(rx1 * sx), int(ry1 * sy)
        x2, y2 = int(rx2 * sx), int(ry2 * sy)
        color = colors[i % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    ann_path = out_dir / "annotated.png"
    cv2.imwrite(str(ann_path), annotated)
    print(f"  annotated -> {ann_path}")


if __name__ == "__main__":
    main()
