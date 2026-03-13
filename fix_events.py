"""Fix events.jsonl: rebase frame numbers and clean up key repeat artifacts.

Usage:
    python fix_events.py <recording_dir> <detection_absolute_tick>

The detection event has a relative frame (from _frame_count), while key events
have absolute ticks. Uses the detection event + known absolute tick to compute
the offset, then cleans up OS key repeat artifacts.
"""

import json
import sys
from pathlib import Path

# Minimum ticks between a key_up and next key_down for same key to count as
# a separate press. Anything shorter is treated as key repeat artifact.
_MIN_GAP_TICKS = 5


def fix(recording_dir: Path, detection_absolute_tick: int) -> None:
    events_path = recording_dir / "events.jsonl"
    meta = json.loads((recording_dir / "meta.json").read_text("utf-8"))
    frame_count = meta["frame_count"]

    events = []
    for line in events_path.read_text("utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))

    # Find detection event to compute offset
    detection = [e for e in events if e["type"] == "detection"]
    if not detection:
        print("ERROR: no detection event found")
        sys.exit(1)

    relative_frame = detection[0]["frame"]
    start_tick = detection_absolute_tick - relative_frame
    print(f"Detection at relative frame {relative_frame}, absolute tick {detection_absolute_tick}")
    print(f"Recording start tick: {start_tick}")

    # Rebase key events
    for e in events:
        if e["type"] in ("key_down", "key_up"):
            e["frame"] = e["frame"] - start_tick

    # Filter out-of-range events
    events = [e for e in events if 0 <= e["frame"] < frame_count]

    # Clean up key artifacts per-key:
    # 1. Remove repeat key_down while key is held
    # 2. Remove spurious short presses after a release (OS key repeat tail)
    # 3. Extend key_up by 1 tick to compensate for sub-frame timing loss
    key_events: dict[str, list] = {}
    non_key = []
    for e in events:
        if e["type"] in ("key_down", "key_up"):
            key = e.get("key", "")
            key_events.setdefault(key, []).append(e)
        else:
            non_key.append(e)

    cleaned = list(non_key)
    for key, kevs in key_events.items():
        kevs.sort(key=lambda e: e["frame"])
        held = False
        last_up_frame = -999
        for e in kevs:
            if e["type"] == "key_down":
                if held:
                    continue  # skip repeat
                if e["frame"] - last_up_frame < _MIN_GAP_TICKS:
                    continue  # too close to previous release, artifact
                held = True
                cleaned.append(e)
            elif e["type"] == "key_up":
                if not held:
                    continue  # orphan
                held = False
                last_up_frame = e["frame"]
                # Extend by 1 tick to compensate for timing granularity loss
                e = dict(e)
                e["frame"] = min(e["frame"] + 1, frame_count - 1)
                cleaned.append(e)

    cleaned.sort(key=lambda e: (e["frame"], e["type"] != "key_down"))

    # Write back
    events_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in cleaned) + "\n",
        encoding="utf-8",
    )

    # Print tab groups for verification
    tab_events = [e for e in cleaned if e.get("key") == "tab"]
    groups = []
    start = None
    for e in tab_events:
        if e["type"] == "key_down" and start is None:
            start = e["frame"]
        elif e["type"] == "key_up" and start is not None:
            groups.append((start, e["frame"], e["frame"] - start))
            start = None

    print(f"\nTab press groups ({len(groups)}):")
    for i, (s, e, dur) in enumerate(groups):
        ok = "OK" if dur >= 6 else "SHORT"
        print(f"  {i+1}: frame {s}-{e} ({dur} ticks = {dur/10:.1f}s) {ok}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <recording_dir> <detection_absolute_tick>")
        sys.exit(1)
    fix(Path(sys.argv[1]), int(sys.argv[2]))
