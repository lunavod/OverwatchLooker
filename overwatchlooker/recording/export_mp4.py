"""Convert a recording (zstd frames) to an MP4 video file."""

import argparse
import sys
from pathlib import Path

import cv2

from overwatchlooker.recording.replay import ReplaySource


def export_mp4(recording_dir: Path, output: Path | None = None) -> Path:
    """Convert a recording directory to an MP4 file.

    Args:
        recording_dir: Path to the recording directory (contains meta.json, frames.bin).
        output: Output MP4 path. Defaults to <recording_dir>.mp4.

    Returns:
        Path to the written MP4 file.
    """
    if output is None:
        output = recording_dir.with_suffix(".mp4")

    src = ReplaySource(recording_dir)
    w, h = src.resolution
    fps = src.fps
    total = src.frame_count

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output}")

    try:
        for i in range(total):
            frame = src.reader.read_next()
            if frame is None:
                break
            writer.write(frame)
            if (i + 1) % (fps * 10) == 0 or i + 1 == total:
                pct = (i + 1) / total * 100
                print(f"\r  {i + 1}/{total} frames ({pct:.0f}%)", end="", flush=True)
        print()
    finally:
        writer.release()
        src.close()

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Saved {output} ({size_mb:.1f} MB, {total} frames, {total/fps:.1f}s)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert recording to MP4")
    parser.add_argument("recording", type=Path, help="Recording directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output MP4 path")
    args = parser.parse_args()

    if not args.recording.is_dir():
        print(f"Error: {args.recording} is not a directory", file=sys.stderr)
        sys.exit(1)

    export_mp4(args.recording, args.output)


if __name__ == "__main__":
    main()
