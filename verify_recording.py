"""Verify that frames.bin contains the correct number of frames per meta.json."""

import json
import struct
import sys
from pathlib import Path


def verify(recording_dir: Path) -> None:
    meta = json.loads((recording_dir / "meta.json").read_text("utf-8"))
    expected = meta["frame_count"]
    fmt = meta.get("format", "zstd")
    w, h = meta["resolution"]
    frame_bytes = w * h * 3

    if fmt == "raw":
        raw_path = recording_dir / "frames.raw"
        if not raw_path.exists():
            print(f"ERROR: frames.raw not found")
            sys.exit(1)
        actual_size = raw_path.stat().st_size
        actual = actual_size // frame_bytes
        remainder = actual_size % frame_bytes
        if remainder:
            print(f"ERROR: frames.raw size {actual_size} is not a multiple of frame size {frame_bytes} (remainder {remainder})")
            sys.exit(1)
    elif fmt == "zstd":
        frames_path = recording_dir / "frames.bin"
        if not frames_path.exists():
            print(f"ERROR: frames.bin not found")
            sys.exit(1)
        actual = 0
        with open(frames_path, "rb") as f:
            while True:
                header = f.read(4)
                if len(header) == 0:
                    break
                if len(header) < 4:
                    print(f"ERROR: truncated header at frame {actual}")
                    sys.exit(1)
                length = struct.unpack("<I", header)[0]
                f.seek(length, 1)  # skip compressed data without reading
                actual += 1
    else:
        print(f"ERROR: unknown format {fmt!r}")
        sys.exit(1)

    if actual == expected:
        print(f"OK: {actual} frames ({meta['duration_seconds']}s at {meta['fps']}fps, {w}x{h}, {fmt})")
    else:
        print(f"MISMATCH: meta says {expected} frames, file has {actual} frames (diff={actual - expected})")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <recording_dir>")
        sys.exit(1)
    verify(Path(sys.argv[1]))
