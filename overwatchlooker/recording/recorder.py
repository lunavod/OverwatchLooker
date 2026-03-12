"""Screen + keyboard recorder using zstd frame compression.

Frames are pushed from the tick loop via push_frame(). The recorder
compresses and writes them in a background thread.
"""

import json
import logging
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import zstandard as zstd

_logger = logging.getLogger("overwatchlooker")

_RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
_TARGET_FPS = 10
_ZSTD_LEVEL = 3  # fast compression; level 3 is a good speed/ratio tradeoff


class Recorder:
    """Records screen frames and keyboard events for later replay.

    Frames are pushed externally via push_frame(). Keyboard events are
    captured via pynput. Compression + I/O happen in a background writer thread.
    """

    def __init__(self):
        self._recording = False
        self._output_dir: Path | None = None
        self._events_file = None
        self._frames_file = None
        self._writer_thread: threading.Thread | None = None
        self._frame_queue: list = []
        self._queue_lock = threading.Lock()
        self._start_time = 0.0
        self._frame_count = 0
        self._resolution: tuple[int, int] = (0, 0)
        self._compressor = zstd.ZstdCompressor(level=_ZSTD_LEVEL)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def output_dir(self) -> Path | None:
        return self._output_dir

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def log_event(self, event_type: str, frame: int | None = None, **kwargs) -> None:
        """Log an event to events.jsonl."""
        if not self._recording or not self._events_file:
            return
        entry = {"frame": frame if frame is not None else self._frame_count,
                 "type": event_type, **kwargs}
        line = json.dumps(entry, ensure_ascii=False)
        try:
            self._events_file.write(line + "\n")
            self._events_file.flush()
        except Exception as e:
            _logger.warning(f"Failed to log event: {e}")

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a BGR frame to be recorded. Called from the tick loop."""
        if not self._recording:
            return
        raw = frame.tobytes()
        with self._queue_lock:
            self._frame_queue.append(raw)
        self._frame_count += 1

    def start(self, resolution: tuple[int, int]) -> Path:
        """Start recording. Returns the output directory path."""
        if self._recording:
            raise RuntimeError("Already recording")

        # Create output directory
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._output_dir = _RECORDINGS_DIR / timestamp
        self._output_dir.mkdir()

        # Open events file
        self._events_file = open(
            self._output_dir / "events.jsonl", "w", encoding="utf-8"
        )

        self._resolution = resolution
        w, h = resolution
        _logger.info(f"Recording at {w}x{h} @ {_TARGET_FPS}fps")

        # Open frames file
        self._frames_file = open(self._output_dir / "frames.bin", "wb")

        self._frame_count = 0
        self._recording = True
        self._start_time = time.monotonic()

        # Start background writer thread (compresses + writes frames)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        _logger.info(f"Recording started: {self._output_dir}")
        return self._output_dir

    def stop(self) -> Path:
        """Stop recording. Returns the output directory path."""
        if not self._recording:
            raise RuntimeError("Not recording")

        self._recording = False
        duration = self._elapsed()

        # Wait for writer to drain queue
        if self._writer_thread:
            self._writer_thread.join(timeout=30.0)

        # Close frames file
        if self._frames_file:
            self._frames_file.close()
            self._frames_file = None

        # Write meta.json
        meta = {
            "start_time": datetime.now().isoformat(),
            "resolution": list(self._resolution),
            "fps": _TARGET_FPS,
            "frame_count": self._frame_count,
            "duration_seconds": round(duration, 1),
            "format": "zstd",
        }
        (self._output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        output = self._output_dir
        frame_count = self._frame_count
        self._cleanup()

        _logger.info(
            f"Recording stopped: {frame_count} frames, "
            f"{duration:.1f}s, saved to {output}"
        )
        return output

    def log_key_events(self, tick: int, pressed: set[str], released: set[str]) -> None:
        """Log key events for a given frame (called by tick loop)."""
        for key in pressed:
            self.log_event("key_down", frame=tick, key=key)
        for key in released:
            self.log_event("key_up", frame=tick, key=key)

    def _writer_loop(self) -> None:
        """Compress and write frame queue to frames.bin file."""
        while self._recording or self._frame_queue:
            batch = None
            with self._queue_lock:
                if self._frame_queue:
                    batch = self._frame_queue
                    self._frame_queue = []
            if batch:
                for raw in batch:
                    try:
                        compressed = self._compressor.compress(raw)
                        self._frames_file.write(
                            struct.pack("<I", len(compressed))
                        )
                        self._frames_file.write(compressed)
                    except Exception as e:
                        _logger.error(f"Frame write failed: {e}")
                        return
            else:
                time.sleep(0.01)

    def _cleanup(self) -> None:
        """Release resources."""
        if self._events_file:
            self._events_file.close()
            self._events_file = None
        if self._frames_file:
            self._frames_file.close()
            self._frames_file = None
        self._frame_count = 0
