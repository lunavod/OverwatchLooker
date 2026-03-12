"""Test hero switch tracking using transcript data.

Simulates the subtitle listener's hero extraction and dedup logic
on a real transcript segment to verify correct switch detection.
"""

import re
import sys
from datetime import datetime


# Import the edit distance function from the subtitle listener
sys.path.insert(0, ".")
from overwatchlooker.subtitle_listener import _edit_distance


def parse_transcript_segment(path: str, start_line: int, end_line: int):
    """Parse transcript lines and extract hero observations with timestamps."""
    observations = []  # (wall_seconds, username, hero)

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[start_line - 1:end_line]:
        line = line.strip()
        # Format: [HH:MM:SS] [username (hero)] dialogue
        m = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s+\[(\w+)\s+\(([^)]+)\)\]", line)
        if not m:
            continue
        ts_str, username, hero = m.group(1), m.group(2).upper(), m.group(3).strip().title()
        if username == "ATHENA":
            continue

        # Convert timestamp to seconds (handle midnight wrap)
        t = datetime.strptime(ts_str, "%H:%M:%S")
        secs = t.hour * 3600 + t.minute * 60 + t.second
        observations.append((secs, username, hero))

    return observations


def simulate_hero_tracking(observations):
    """Simulate the subtitle listener's hero tracking with dedup."""
    hero_history: dict[str, list[tuple[float, str]]] = {}

    for secs, username, hero in observations:
        history = hero_history.get(username)
        if history:
            last_hero = history[-1][1]
            dist = _edit_distance(hero.lower(), last_hero.lower())
            if dist <= 2:
                continue  # OCR artifact
        if username not in hero_history:
            hero_history[username] = []
        hero_history[username].append((secs, hero))

    return hero_history


def format_switches(hero_history):
    """Format hero switches for display, matching the format_match output."""
    all_times = [t for entries in hero_history.values() for t, _ in entries]
    if not all_times:
        return ""

    match_start = min(all_times)
    lines = []

    for username in sorted(hero_history):
        entries = hero_history[username]
        if len(entries) <= 1:
            hero = entries[0][1] if entries else "?"
            lines.append(f"  {username}: {hero} (no switches)")
            continue

        parts = []
        for i, (t, hero) in enumerate(entries):
            start_s = int(t - match_start)
            start_mm, start_ss = divmod(start_s, 60)
            if i < len(entries) - 1:
                end_s = int(entries[i + 1][0] - match_start)
                end_mm, end_ss = divmod(end_s, 60)
                parts.append(f"{hero} ({start_mm}:{start_ss:02d}-{end_mm}:{end_ss:02d})")
            else:
                parts.append(f"{hero} ({start_mm}:{start_ss:02d}+)")
        lines.append(f"  {username}: {', '.join(parts)}")

    return "\n".join(lines)


if __name__ == "__main__":
    transcript = "transcripts/2026-03-11_21-27-15.txt"

    # Last defeat match: lines 2606-3069
    print("=== Last match (lines 2606-3069) ===")
    print()

    obs = parse_transcript_segment(transcript, 2606, 3069)
    print(f"Total hero observations: {len(obs)}")
    print(f"Unique players: {len(set(u for _, u, _ in obs))}")
    print()

    history = simulate_hero_tracking(obs)

    # Show raw unique hero readings per player (before dedup)
    raw_heroes: dict[str, set[str]] = {}
    for _, username, hero in obs:
        raw_heroes.setdefault(username, set()).add(hero)

    print("--- Raw OCR readings per player ---")
    for u in sorted(raw_heroes):
        heroes = sorted(raw_heroes[u])
        print(f"  {u}: {heroes}")

    print()
    print("--- Detected hero switches (with dedup) ---")
    print(format_switches(history))

    # Show which OCR artifacts were filtered
    print()
    print("--- Filtered OCR artifacts ---")
    for username in sorted(raw_heroes):
        raw = raw_heroes[username]
        tracked = {h for _, h in history.get(username, [])}
        filtered = raw - tracked
        if filtered:
            canonical = tracked
            print(f"  {username}: {sorted(filtered)} merged into {sorted(canonical)}")
