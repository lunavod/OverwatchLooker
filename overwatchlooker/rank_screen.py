"""Rank screen analysis system — captures frames after match_ended and OCRs rank progression.

Purely tick-based: no wall-clock timing. Works identically in live and replay modes.
Activated on MatchEndEvent for ranked matches (game_type == RANKED). Collects frames
for ~10s (delta window), grabs main frame at ~10s, then runs OCR in a background thread.

Replay compatibility:
    The system checks ms.game_type == RANKED to decide whether to activate.
    When replaying with --replay-start, game_type is available because TickLoop
    flushes all Overwolf events up to the start tick before the first real tick
    runs (see tick.py seek logic). GameTypeUpdate fires at frame 0, so it's
    always processed even when starting from the middle of a recording.
"""

import logging
import threading

import numpy as np

from overwatchlooker.rank_ocr import (
    REGIONS,
    _REF_W, _REF_H,
    _RANK_MODEL_DIR,
    _VALUES_MODEL_DIR,
    _MODIFIERS_MODEL_DIR,
    _COLOR_TOLERANCE,
    _scale_region,
    binarize_bbox,
    extract_progress_bar_delta,
    DELTA_GREEN_BGR_1,
    DELTA_GREEN_BGR_2,
    DELTA_RED_BGR,
    ocr_rank_division,
    ocr_rank_progress,
    ocr_progress_bar,
    ocr_modifiers,
)
from overwatchlooker.tick import TickContext

_logger = logging.getLogger("overwatchlooker")

_DELTA_WINDOW_SEC = 10.0
_MAIN_TIME_SEC = 10.0


class RankScreenSystem:
    """Captures frames after match_ended and extracts rank progression via OCR."""

    def __init__(self, fps: int = 10) -> None:
        self._fps = fps
        self._delta_window_ticks = int(_DELTA_WINDOW_SEC * fps)
        self._main_time_tick = int(_MAIN_TIME_SEC * fps)

        self._active = False
        self._start_tick = 0
        self._match_state: object | None = None

        # Frame collection
        self._delta_candidates: list[tuple[int, np.ndarray]] = []
        self._main_frame: np.ndarray | None = None

        # Scaled progress bar region (computed from first frame)
        self._pb_region: tuple[int, int, int, int] | None = None
        self._pixel_threshold = 200

        # Models (loaded lazily, cached across matches)
        self._rank_model = None
        self._values_model = None
        self._modifiers_model = None

        # Completion signal
        self._done = threading.Event()
        self._done.set()  # not active = already done

    @property
    def active(self) -> bool:
        return self._active

    def start(self, tick: int, match_state: object) -> None:
        """Activate frame collection starting from this tick."""
        self._active = True
        self._start_tick = tick
        self._match_state = match_state
        self._done.clear()
        self._delta_candidates = []
        self._main_frame = None
        self._pb_region = None
        _logger.info(f"Rank screen: started at tick {tick}, "
                     f"collecting {self._delta_window_ticks} ticks for delta, "
                     f"main frame at tick +{self._main_time_tick}")

    def on_tick(self, ctx: TickContext) -> None:
        if not self._active:
            return

        elapsed = ctx.tick - self._start_tick
        frame = ctx.frame_bgr

        # Compute scaled region on first frame
        if self._pb_region is None:
            h, w = frame.shape[:2]
            sx, sy = w / _REF_W, h / _REF_H
            pb = REGIONS["progress_bar"]
            self._pb_region = (
                int(pb[0] * sx), int(pb[1] * sy),
                int(pb[2] * sx), int(pb[3] * sy))
            self._pixel_threshold = int(200 * sx * sy)

        # Check for delta bar pixels during delta window
        if elapsed <= self._delta_window_ticks:
            self._check_delta_candidate(elapsed, frame)

        # Capture main frame
        if elapsed == self._main_time_tick:
            self._main_frame = frame.copy()

        # Done collecting — launch analysis in background
        if elapsed == self._main_time_tick + 1:
            self._active = False
            threading.Thread(
                target=self._run_analysis, daemon=True).start()

    def _check_delta_candidate(self, tick_offset: int,
                               frame: np.ndarray) -> None:
        """Check if frame has green/red delta bar pixels."""
        x1, y1, x2, y2 = self._pb_region  # type: ignore[misc]
        crop = frame[y1:y2, x1:x2]
        img_f = crop.astype(float)

        gc = int(np.sum(
            (np.sqrt(np.sum((img_f - DELTA_GREEN_BGR_1) ** 2, axis=2)) < _COLOR_TOLERANCE) |
            (np.sqrt(np.sum((img_f - DELTA_GREEN_BGR_2) ** 2, axis=2)) < _COLOR_TOLERANCE)))
        rc = int(np.sum(
            np.sqrt(np.sum((img_f - DELTA_RED_BGR) ** 2, axis=2)) < _COLOR_TOLERANCE))

        if gc + rc > self._pixel_threshold:
            self._delta_candidates.append((tick_offset, frame.copy()))

    def _load_models(self) -> None:
        """Load OCR models (once, cached across matches)."""
        if self._rank_model is not None:
            return
        from paddlex import create_model  # type: ignore[import-untyped]
        _logger.info("Rank screen: loading OCR models...")
        self._rank_model = create_model(
            model_name="PP-OCRv5_server_rec",
            model_dir=str(_RANK_MODEL_DIR))
        self._values_model = create_model(
            model_name="PP-OCRv5_server_rec",
            model_dir=str(_VALUES_MODEL_DIR))
        self._modifiers_model = create_model(
            model_name="PP-OCRv5_server_rec",
            model_dir=str(_MODIFIERS_MODEL_DIR))
        _logger.info("Rank screen: models loaded")

    @staticmethod
    def _parse_pct(text: str, sign: str = "+") -> int:
        """Parse OCR percentage text like '27%' into a signed int."""
        cleaned = text.replace("%", "").strip()
        try:
            val = int(cleaned)
        except ValueError:
            return 0
        return val if sign == "+" else -val

    @staticmethod
    def _parse_rank_division(text: str) -> tuple[str, int]:
        """Parse OCR rank text like 'GOLD 3' into ('GOLD', 3)."""
        parts = text.strip().split()
        if len(parts) >= 2:
            try:
                return parts[0].upper(), int(parts[-1])
            except ValueError:
                pass
        return text.upper(), 0

    def _run_analysis(self) -> None:
        """Run OCR on collected frames, store results in MatchState."""
        try:
            import cv2
            from pathlib import Path

            self._load_models()

            # Debug output directory — overwritten each match
            dbg = Path(__file__).parent.parent / "debug_rank"
            dbg.mkdir(exist_ok=True)

            from overwatchlooker.match_state import RankProgression
            result = RankProgression()

            # Find delta from candidates (try last 10 in reverse)
            delta_text = None
            delta_sign = None
            _logger.info(f"Rank screen: {len(self._delta_candidates)} delta candidates")
            if self._delta_candidates:
                # Save all candidate frames
                for i, (tick_off, frame) in enumerate(self._delta_candidates[-10:]):
                    cv2.imwrite(str(dbg / f"delta_candidate_{i}_tick{tick_off}.png"), frame)

                for tick_off, frame in reversed(self._delta_candidates[-10:]):
                    dt, ds, dsign = ocr_progress_bar(frame, self._values_model)
                    if dt is not None and ds > 0.3:
                        delta_text = dt
                        delta_sign = dsign
                        _logger.info(f"Rank screen: delta={dsign}{dt} "
                                     f"(score={ds:.4f}, tick +{tick_off})")
                        # Save winning candidate details
                        cv2.imwrite(str(dbg / "delta_frame.png"), frame)
                        h, w = frame.shape[:2]
                        sx, sy = w / _REF_W, h / _REF_H
                        scale = (sx + sy) / 2
                        x1, y1, x2, y2 = _scale_region("progress_bar", sx, sy)
                        crop = frame[y1:y2, x1:x2]
                        cv2.imwrite(str(dbg / "delta_pb_crop.png"), crop)
                        ocr_img, _ = extract_progress_bar_delta(crop, scale)
                        if ocr_img is not None:
                            cv2.imwrite(str(dbg / "delta_ocr_input.png"), ocr_img)
                        break

            # Main frame analysis
            main = self._main_frame
            if main is None:
                _logger.warning("Rank screen: no main frame captured")
                return

            # Check for demotion protection on main frame
            main_dt, main_ds, main_dsign = ocr_progress_bar(
                main, self._values_model)
            if main_dt is None and main_dsign is None:
                result.demotion_protection = True
                _logger.info("Rank screen: demotion protection detected")
            elif delta_text is None and main_dt:
                delta_text = main_dt
                delta_sign = main_dsign

            if delta_text and delta_sign:
                result.delta_pct = self._parse_pct(delta_text, delta_sign)

            # Rank + division
            rank_text, rank_score = ocr_rank_division(
                main, self._rank_model)
            result.rank, result.division = self._parse_rank_division(rank_text)
            _logger.info(f"Rank screen: rank={result.rank} {result.division} "
                         f"(score={rank_score:.4f})")

            # Progress
            prog_text, prog_score, prog_sign = ocr_rank_progress(
                main, self._values_model)
            result.progress_pct = self._parse_pct(prog_text, prog_sign)
            _logger.info(f"Rank screen: progress={result.progress_pct}% "
                         f"(score={prog_score:.4f})")

            # Modifiers
            mods = ocr_modifiers(main, self._modifiers_model)
            result.modifiers = [m[0] for m in mods]
            if mods:
                _logger.info(f"Rank screen: modifiers="
                             f"{', '.join(result.modifiers)}")

            # Store in match state
            if self._match_state:
                self._match_state.rank_progression = result  # type: ignore[attr-defined]

            _logger.info("Rank screen: analysis complete")

            # Debug: save all images for offline debugging (never crashes analysis)
            try:
                h, w = main.shape[:2]
                sx, sy = w / _REF_W, h / _REF_H
                cv2.imwrite(str(dbg / "main_frame.png"), main)
                for region_name in ("rank_division", "rank_progress",
                                    "progress_bar", "modifiers"):
                    rx1, ry1, rx2, ry2 = _scale_region(region_name, sx, sy)
                    cv2.imwrite(str(dbg / f"main_{region_name}.png"),
                                main[ry1:ry2, rx1:rx2])
                # Rank binarized input
                rx1, ry1, rx2, ry2 = _scale_region("rank_division", sx, sy)
                rank_gray = cv2.cvtColor(main[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(dbg / "rank_ocr_input.png"),
                            binarize_bbox(rank_gray))
                # Progress binarized input
                px1, py1, px2, py2 = _scale_region("rank_progress", sx, sy)
                prog_gray = cv2.cvtColor(main[py1:py2, px1:px2], cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(dbg / "progress_ocr_input.png"),
                            binarize_bbox(prog_gray))
            except Exception as e:
                _logger.debug(f"Debug image save failed: {e}")

        except Exception as e:
            _logger.warning(f"Rank screen analysis failed: {e}", exc_info=True)
        finally:
            self._done.set()

    def wait_done(self, timeout: float = 30.0) -> bool:
        """Wait for analysis to complete. Returns True if completed."""
        return self._done.wait(timeout=timeout)
