"""Replay a recorded session from MP4 video + .meta keyboard data."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from memoir_capture import MetaReader, MetaFile

_logger = logging.getLogger("overwatchlooker")



class FrameReader:
    """Reads frames sequentially from an MP4 via cv2.VideoCapture."""

    def __init__(self, video_path: Path):
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._read = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read_next(self) -> np.ndarray | None:
        """Read next frame sequentially. Returns BGR ndarray or None when exhausted."""
        if self._read >= self._frame_count:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._read += 1
        return frame

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame index."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._read = frame_index

    def close(self) -> None:
        self._cap.release()


def _synthesize_events(meta: MetaFile) -> list[dict]:
    """Convert keyboard_mask diffs between consecutive meta rows into key events."""
    events: list[dict] = []
    prev_keys: set[str] = set()
    for row in meta:
        current = set(row.pressed_keys(meta.keys))
        for key in current - prev_keys:
            events.append({"frame": row.record_frame_index,
                           "type": "key_down", "key": key})
        for key in prev_keys - current:
            events.append({"frame": row.record_frame_index,
                           "type": "key_up", "key": key})
        prev_keys = current

    events.sort(key=lambda e: e["frame"])
    return events


class ReplaySource:
    """Loads an MP4 recording + .meta and provides frame access + event scheduling."""

    def __init__(self, source: Path | str):
        """
        Args:
            source: Path to a recording directory (containing recording.mp4 + recording.meta),
                    or a direct .mp4 file path.
        """
        source = Path(source)

        if source.is_dir():
            video_path = source / "recording.mp4"
            meta_path = source / "recording.meta"
            overwolf_path = source / "recording.overwolf.jsonl"
        elif source.suffix == ".mp4":
            video_path = source
            meta_path = source.with_suffix(".meta")
            overwolf_path = source.with_suffix(".overwolf.jsonl")
        else:
            raise FileNotFoundError(f"Cannot determine recording format from: {source}")

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video to get properties
        self._reader = FrameReader(video_path)

        cap = cv2.VideoCapture(str(video_path))
        self._fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
        self._frame_count = self._reader.frame_count
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (w, h)
        self._duration = self._frame_count / self._fps if self._fps else 0.0
        cap.release()

        # Load .meta if available
        self._meta: MetaFile | None = None
        self._events: list[dict] = []
        if meta_path.exists():
            self._meta = MetaReader.read(meta_path)
            self._events = _synthesize_events(self._meta)
            _logger.info(f"Loaded .meta: {len(self._meta.rows)} rows, "
                         f"{len(self._events)} synthetic key events")
        else:
            _logger.warning(f"No .meta file found at {meta_path}, replay without keyboard data")

        # Overwolf events path (may or may not exist)
        self._overwolf_events_path: Path | None = overwolf_path if overwolf_path.exists() else None
        if self._overwolf_events_path:
            _logger.info(f"Found Overwolf events: {overwolf_path}")

        _logger.info(
            f"Replay loaded: {self._frame_count} frames, {self._duration:.1f}s, "
            f"{self._resolution[0]}x{self._resolution[1]}"
        )

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def reader(self) -> FrameReader:
        return self._reader

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    @property
    def overwolf_events_path(self) -> Path | None:
        return self._overwolf_events_path

    def close(self) -> None:
        """Release resources."""
        self._reader.close()
