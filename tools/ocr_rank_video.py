"""Extract rank progression data from a post-match rank screen video.

Scans the first N seconds for the delta frame (last frame with red/green
delta bar), then uses a later frame for everything else (rank, progress,
modifiers, demotion protection).

Usage:
    uv run python tools/ocr_rank_video.py <video.mp4> [--delta-window 7] [--main-time 10]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from paddlex import create_model

from crop_rank_screen import (
    REGIONS,
    _REF_W, _REF_H,
    _RANK_MODEL_DIR,
    _VALUES_MODEL_DIR,
    _MODIFIERS_MODEL_DIR,
    _COLOR_TOLERANCE,
    ocr_rank_division,
    ocr_rank_progress,
    ocr_progress_bar,
    ocr_modifiers,
    extract_progress_bar_delta,
)

# Delta bar colors in BGR
_GREEN_BGR_1 = np.array([36, 234, 76], dtype=float)
_GREEN_BGR_2 = np.array([0, 252, 3], dtype=float)
_RED_BGR = np.array([95, 47, 181], dtype=float)


def find_delta_frame(cap: cv2.VideoCapture, fps: float, window_sec: float,
                     values_model) -> tuple[np.ndarray | None, int, str | None, str | None, float]:
    """Scan first N seconds for a frame where the delta can be read.

    Collects frames with significant delta bar pixels, then tries OCR
    on the last few candidates (reverse order) until one succeeds.

    Returns (frame, frame_index, delta_text, delta_sign, score) or
    (None, -1, None, None, 0) if not found.
    """
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = frame_w / _REF_W, frame_h / _REF_H

    pb = REGIONS["progress_bar"]
    pb_x1, pb_y1 = int(pb[0] * sx), int(pb[1] * sy)
    pb_x2, pb_y2 = int(pb[2] * sx), int(pb[3] * sy)

    max_frames = int(window_sec * fps)
    tolerance = _COLOR_TOLERANCE
    pixel_threshold = int(200 * sx * sy)

    # Collect frames with significant delta pixels
    candidates: list[tuple[int, np.ndarray]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[pb_y1:pb_y2, pb_x1:pb_x2]
        img_f = crop.astype(float)
        gc = int(np.sum(
            (np.sqrt(np.sum((img_f - _GREEN_BGR_1) ** 2, axis=2)) < tolerance) |
            (np.sqrt(np.sum((img_f - _GREEN_BGR_2) ** 2, axis=2)) < tolerance)))
        rc = int(np.sum(np.sqrt(np.sum((img_f - _RED_BGR) ** 2, axis=2)) < tolerance))
        if gc + rc > pixel_threshold:
            candidates.append((i, frame.copy()))

    if not candidates:
        return None, -1, None, None, 0.0

    # Try OCR on the last N candidates in reverse order
    for idx, frame in reversed(candidates[-10:]):
        dt, ds, dsign = ocr_progress_bar(frame, values_model)
        if dt is not None and ds > 0.3:
            return frame, idx, dt, dsign, ds

    return None, -1, None, None, 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Extract rank progression from post-match video")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--delta-window", type=float, default=10.0,
                        help="Seconds to scan for delta frame (default: 10)")
    parser.add_argument("--main-time", type=float, default=10.0,
                        help="Seconds into video for main frame (default: 10)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    print(f"Video: {video_path.name} ({duration:.1f}s, {fps:.0f} FPS, "
          f"{total_frames} frames, {frame_w}x{frame_h})")

    # Load models
    print("Loading models...")
    rank_model = create_model(model_name="PP-OCRv5_server_rec",
                              model_dir=str(_RANK_MODEL_DIR))
    values_model = create_model(model_name="PP-OCRv5_server_rec",
                                model_dir=str(_VALUES_MODEL_DIR))
    modifiers_model = create_model(model_name="PP-OCRv5_server_rec",
                                   model_dir=str(_MODIFIERS_MODEL_DIR))

    # Find delta frame
    print(f"\nScanning first {args.delta_window}s for delta frame...")
    delta_frame, delta_idx, delta_text, delta_sign, delta_score = \
        find_delta_frame(cap, fps, args.delta_window, values_model)

    demotion_protection = False

    if delta_text is not None:
        print(f"  Found delta at frame {delta_idx} ({delta_idx/fps:.1f}s)")
        print(f"  Delta: {delta_sign}{delta_text} ({delta_score:.4f})")
    elif delta_idx >= 0:
        print(f"  Delta bar found at frame {delta_idx} but could not read text")
    else:
        print("  No delta frame found")

    # Read main frame
    main_frame_idx = int(args.main_time * fps)
    if main_frame_idx >= total_frames:
        main_frame_idx = total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, main_frame_idx)
    ok, main_frame = cap.read()
    cap.release()

    if not ok:
        print(f"Failed to read frame at {args.main_time}s", file=sys.stderr)
        sys.exit(1)

    print(f"\nMain frame at {main_frame_idx/fps:.1f}s:")

    # Rank + division
    rank_text, rank_score = ocr_rank_division(main_frame, rank_model)
    print(f"  Rank: {rank_text} ({rank_score:.4f})")

    # Progress
    prog_text, prog_score, prog_sign = ocr_rank_progress(main_frame, values_model)
    print(f"  Progress: {prog_sign}{prog_text} ({prog_score:.4f})")

    # Check main frame for demotion protection
    main_dt, main_ds, main_dsign = ocr_progress_bar(main_frame, values_model)
    if main_dt is None and main_dsign is None:
        demotion_protection = True
        print(f"  Delta: DEMOTION PROTECTION")
    elif delta_text is None:
        # No delta from earlier scan, use main frame
        delta_text = main_dt
        delta_sign = main_dsign
        if delta_text:
            print(f"  Delta: {delta_sign}{delta_text} ({main_ds:.4f})")

    # Modifiers
    mods = ocr_modifiers(main_frame, modifiers_model)
    mod_names = [m[0] for m in mods]
    if mods:
        print(f"  Modifiers: {', '.join(mod_names)}")
    else:
        print(f"  Modifiers: none")

    # Summary
    print(f"\n{'═' * 40}")
    print(f"  Rank:       {rank_text}")
    print(f"  Progress:   {prog_sign}{prog_text}")
    if delta_text:
        print(f"  Delta:      {delta_sign}{delta_text}")
    if demotion_protection:
        print(f"  Demotion:   PROTECTION")
    if mod_names:
        print(f"  Modifiers:  {', '.join(mod_names)}")
    print(f"{'═' * 40}")


if __name__ == "__main__":
    main()
