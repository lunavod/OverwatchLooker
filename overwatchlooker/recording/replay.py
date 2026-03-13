"""Replay a recorded session, feeding frames and events to the app pipeline."""

import json
import logging
import struct
from pathlib import Path

import cv2
import numpy as np
import zstandard as zstd

_logger = logging.getLogger("overwatchlooker")


def _ensure_raw_frames(recording_dir: Path, meta: dict) -> Path:
    """Decompress frames.bin → frames.raw if not already done. Returns path to raw file."""
    raw_path = recording_dir / "frames.raw"
    frames_path = recording_dir / "frames.bin"

    if raw_path.exists():
        frame_size = meta["resolution"][0] * meta["resolution"][1] * 3
        actual_size = raw_path.stat().st_size
        if actual_size == 0 or actual_size % frame_size != 0:
            raise RuntimeError(
                f"frames.raw corrupt: size={actual_size} not a multiple of frame_size={frame_size}. "
                f"Delete frames.raw manually to regenerate."
            )
        actual_frames = actual_size // frame_size
        if actual_frames != meta["frame_count"]:
            _logger.warning(
                f"frames.raw has {actual_frames} frames, meta says {meta['frame_count']}. "
                f"Using actual count."
            )
            meta["frame_count"] = actual_frames
        return raw_path

    if not frames_path.exists():
        raise FileNotFoundError(f"No frames.bin in {recording_dir}")

    w, h = meta["resolution"]
    frame_bytes = h * w * 3
    frame_count = meta["frame_count"]
    print(f"Decompressing {frame_count} frames to frames.raw...")

    decompressor = zstd.ZstdDecompressor()
    with open(frames_path, "rb") as src, open(raw_path, "wb") as dst:
        for i in range(frame_count):
            header = src.read(4)
            if len(header) < 4:
                break
            length = struct.unpack("<I", header)[0]
            compressed = src.read(length)
            if len(compressed) < length:
                break
            raw = decompressor.decompress(compressed)
            dst.write(raw)

    actual_frames = raw_path.stat().st_size // frame_bytes
    if actual_frames != frame_count:
        _logger.warning(f"Decompressed {actual_frames} frames (meta says {frame_count}), using actual count")
        meta["frame_count"] = actual_frames
    _logger.info(f"Decompressed to {raw_path} ({raw_path.stat().st_size / 1024 / 1024:.0f} MB)")
    return raw_path


class FrameReader:
    """Reads frames sequentially from memory-mapped raw frames, zstd, or video."""

    def __init__(self, recording_dir: Path, meta: dict, no_cache: bool = False):
        self._resolution = tuple(meta["resolution"])
        w, h = self._resolution
        self._frame_shape = (h, w, 3)
        self._frame_count = meta["frame_count"]
        self._fps = meta["fps"]
        self._read = 0
        self._frame_bytes = h * w * 3

        raw_path = recording_dir / "frames.raw"
        frames_path = recording_dir / "frames.bin"
        video_path = recording_dir / "video.mkv"

        # Priority: raw file > zstd (decompress to raw first) > video
        if raw_path.exists():
            self._validate_raw(raw_path, meta)
            self._mode = "raw"
            self._raw_file = open(raw_path, "rb")
            self._frame_count = meta["frame_count"]
            self._file = None
            self._decompressor = None
            self._cap = None
        elif frames_path.exists() and not no_cache:
            _ensure_raw_frames(recording_dir, meta)
            self._mode = "raw"
            self._raw_file = open(raw_path, "rb")
            self._frame_count = meta["frame_count"]
            self._file = None
            self._decompressor = None
            self._cap = None
        elif frames_path.exists():
            self._mode = "zstd"
            self._file = open(frames_path, "rb")
            self._decompressor = zstd.ZstdDecompressor()
            self._raw_file = None
            self._cap = None
        elif video_path.exists():
            self._mode = "video"
            self._cap = cv2.VideoCapture(str(video_path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open {video_path}")
            self._file = None
            self._decompressor = None
            self._raw_file = None
        else:
            raise FileNotFoundError(
                f"No frames.bin or video.mkv in {recording_dir}"
            )

        _logger.info(f"Frame reader: {self._mode} format")

    def _validate_raw(self, raw_path: Path, meta: dict) -> None:
        """Validate frames.raw and adjust meta frame_count if needed."""
        frame_size = meta["resolution"][0] * meta["resolution"][1] * 3
        actual_size = raw_path.stat().st_size
        if actual_size == 0 or actual_size % frame_size != 0:
            raise RuntimeError(
                f"frames.raw corrupt: size={actual_size} not a multiple of frame_size={frame_size}. "
                f"Delete frames.raw manually to regenerate."
            )
        actual_frames = actual_size // frame_size
        if actual_frames != meta["frame_count"]:
            _logger.warning(
                f"frames.raw has {actual_frames} frames, meta says {meta['frame_count']}. "
                f"Using actual count."
            )
            meta["frame_count"] = actual_frames

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read_next(self) -> np.ndarray | None:
        """Read next frame sequentially. Returns None when exhausted."""
        if self._read >= self._frame_count:
            return None

        if self._mode == "raw":
            frame = self._read_raw()
        elif self._mode == "zstd":
            frame = self._read_zstd()
        else:
            frame = self._read_video()

        self._read += 1
        return frame

    def _read_raw(self) -> np.ndarray | None:
        data = self._raw_file.read(self._frame_bytes)
        if len(data) < self._frame_bytes:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(self._frame_shape)

    def _read_zstd(self) -> np.ndarray | None:
        header = self._file.read(4)
        if len(header) < 4:
            return None
        length = struct.unpack("<I", header)[0]
        compressed = self._file.read(length)
        if len(compressed) < length:
            return None
        raw = self._decompressor.decompress(compressed)
        return np.frombuffer(raw, dtype=np.uint8).reshape(self._frame_shape)

    def _read_video(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        if hasattr(self, '_raw_file') and self._raw_file:
            self._raw_file.close()
            self._raw_file = None
        if self._file:
            self._file.close()
            self._file = None
        if self._cap:
            self._cap.release()
            self._cap = None


class ReplaySource:
    """Loads a recording and provides frame access + event scheduling."""

    def __init__(self, recording_dir: Path, no_cache: bool = False):
        self._dir = recording_dir

        # Load meta
        meta_path = recording_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No meta.json in {recording_dir}")
        self._meta = json.loads(meta_path.read_text("utf-8"))
        self._fps = self._meta["fps"]
        self._frame_count = self._meta["frame_count"]
        self._resolution = tuple(self._meta["resolution"])
        self._duration = self._meta["duration_seconds"]

        # Frame reader (handles both zstd and video formats)
        self._reader = FrameReader(recording_dir, self._meta, no_cache=no_cache)

        # Load events
        events_path = recording_dir / "events.jsonl"
        self._events = []
        if events_path.exists():
            for line in events_path.read_text("utf-8").splitlines():
                line = line.strip()
                if line:
                    self._events.append(json.loads(line))
            self._events.sort(key=lambda e: e.get("frame", 0))

        _logger.info(
            f"Replay loaded: {self._frame_count} frames, {self._duration:.1f}s, "
            f"{self._resolution[0]}x{self._resolution[1]}"
        )

    @property
    def meta(self) -> dict:
        return dict(self._meta)

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

    def close(self) -> None:
        """Release resources."""
        self._reader.close()
