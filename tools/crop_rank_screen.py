"""Crop regions of interest from the post-match rank screen and OCR them.

CLI tool for debugging and visualization. Core OCR logic lives in
overwatchlooker.rank_ocr.

Usage:
    uv run python tools/crop_rank_screen.py <frame.png>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from paddlex import create_model

from overwatchlooker.rank_ocr import (
    REGIONS,
    _REF_W, _REF_H,
    _RANK_MODEL_DIR,
    _VALUES_MODEL_DIR,
    _MODIFIERS_MODEL_DIR,
    _scale_region,
    binarize_bbox,
    isolate_colored_text,
    extract_progress_bar_delta,
    ocr_rank_division,
    ocr_rank_progress,
    ocr_progress_bar,
    ocr_modifiers,
)


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
        print("  Delta: DEMOTION PROTECTION")
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
        print("  Modifiers: none")

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
