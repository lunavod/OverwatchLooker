"""Create a trimmed replay containing only frames around key events (e.g. tab presses).

Usage:
    python trim_replay.py <recording_dir> [--key tab] [--padding 20] [--out <output_dir>]

Copies only frames within ±padding of any key event for the specified key,
plus detection events. Creates a new recording directory with trimmed frames.raw,
adjusted events.jsonl, and updated meta.json.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np


def trim(recording_dir: Path, key: str = "tab", padding: int = 20,
         output_dir: Path | None = None) -> None:
    meta = json.loads((recording_dir / "meta.json").read_text("utf-8"))
    frame_count = meta["frame_count"]
    w, h = meta["resolution"]
    frame_bytes = w * h * 3

    events = []
    events_path = recording_dir / "events.jsonl"
    for line in events_path.read_text("utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))

    # Find frames to keep: around key events + detection
    key_frames = set()
    for e in events:
        if e.get("key") == key or e.get("type") == "detection":
            f = e["frame"]
            for i in range(max(0, f - padding), min(frame_count, f + padding + 1)):
                key_frames.add(i)

    keep = sorted(key_frames)
    if not keep:
        print(f"No {key} events found")
        sys.exit(1)

    # Build old->new frame mapping
    old_to_new = {old: new for new, old in enumerate(keep)}

    # Remap events
    new_events = []
    for e in events:
        f = e["frame"]
        if f in old_to_new:
            e = dict(e)
            e["frame"] = old_to_new[f]
            new_events.append(e)

    # Determine source format
    raw_path = recording_dir / "frames.raw"
    bin_path = recording_dir / "frames.bin"

    if output_dir is None:
        output_dir = recording_dir.parent / (recording_dir.name + "_trimmed")
    output_dir.mkdir(exist_ok=True)

    print(f"Keeping {len(keep)} of {frame_count} frames ({len(keep)/frame_count*100:.1f}%)")

    if raw_path.exists():
        # Read from raw
        out_raw = output_dir / "frames.raw"
        with open(raw_path, "rb") as src, open(out_raw, "wb") as dst:
            for old_frame in keep:
                src.seek(old_frame * frame_bytes)
                data = src.read(frame_bytes)
                dst.write(data)
    elif bin_path.exists():
        # Read from zstd, write as raw
        import zstandard as zstd
        decompressor = zstd.ZstdDecompressor()
        out_raw = output_dir / "frames.raw"
        keep_set = set(keep)
        with open(bin_path, "rb") as src, open(out_raw, "wb") as dst:
            for i in range(frame_count):
                header = src.read(4)
                if len(header) < 4:
                    break
                length = struct.unpack("<I", header)[0]
                compressed = src.read(length)
                if i in keep_set:
                    raw = decompressor.decompress(compressed)
                    dst.write(raw)
    else:
        print("ERROR: no frames.raw or frames.bin found")
        sys.exit(1)

    # Write meta
    new_meta = dict(meta)
    new_meta["frame_count"] = len(keep)
    new_meta["format"] = "raw"
    new_meta["duration_seconds"] = round(len(keep) / meta["fps"], 1)
    new_meta["trimmed_from"] = str(recording_dir.name)
    (output_dir / "meta.json").write_text(
        json.dumps(new_meta, indent=2), encoding="utf-8"
    )

    # Write events
    (output_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in new_events) + "\n",
        encoding="utf-8",
    )

    size_mb = (len(keep) * frame_bytes) / 1024 / 1024
    print(f"Trimmed replay: {output_dir} ({len(keep)} frames, {size_mb:.0f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--key", default="tab")
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    trim(args.recording_dir, key=args.key, padding=args.padding, output_dir=args.out)
