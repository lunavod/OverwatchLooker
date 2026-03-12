"""Shared schema, formatting, and cost logging for all analyzers."""

import datetime
import json
import logging
from pathlib import Path

_logger = logging.getLogger("overwatchlooker")

_COST_LOG = Path(__file__).parent.parent.parent / "api_costs.jsonl"

# JSON schema for structured output — matches OverwatchStatsMCP submit_match format
MATCH_SCHEMA = {
    "name": "match_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "not_ow2_tab": {
                "type": "boolean",
                "description": "True if the screenshot is NOT an Overwatch 2 Tab screen.",
            },
            "map_name": {
                "type": "string",
                "description": "Map name (e.g. 'Lijiang Tower').",
            },
            "duration": {
                "type": "string",
                "description": "Match duration as MM:SS (e.g. '8:06').",
            },
            "mode": {
                "type": "string",
                "enum": ["PUSH", "CONTROL", "ESCORT", "HYBRID", "CLASH", "FLASHPOINT"],
                "description": "Game mode. Strip any '- COMPETITIVE' suffix.",
            },
            "queue_type": {
                "type": "string",
                "enum": ["COMPETITIVE", "QUICKPLAY"],
                "description": "COMPETITIVE if mode text includes '- COMPETITIVE', else QUICKPLAY.",
            },
            "result": {
                "type": "string",
                "enum": ["VICTORY", "DEFEAT", "UNKNOWN"],
                "description": "Match result from ATHENA subtitle text, audio hint, or UNKNOWN.",
            },
            "players": {
                "type": "array",
                "description": "All 10 players from both teams.",
                "items": {
                    "type": "object",
                    "properties": {
                        "team": {
                            "type": "string",
                            "enum": ["ALLY", "ENEMY"],
                            "description": "ALLY = your team, ENEMY = opposing team.",
                        },
                        "role": {
                            "type": "string",
                            "enum": ["TANK", "DPS", "SUPPORT"],
                            "description": "Player role from the role icon.",
                        },
                        "player_name": {
                            "type": "string",
                            "description": "Player BattleTag in UPPERCASE exactly as shown on screen.",
                        },
                        "title": {
                            "type": ["string", "null"],
                            "description": "Title shown below the player name in Title Case (e.g. 'Stalwart Hero', 'Medic', 'Data Broker'). null if not visible.",
                        },
                        "eliminations": {
                            "type": ["integer", "null"],
                            "description": "Eliminations (E column). null if unreadable.",
                        },
                        "assists": {
                            "type": ["integer", "null"],
                            "description": "Assists (A column). null if unreadable.",
                        },
                        "deaths": {
                            "type": ["integer", "null"],
                            "description": "Deaths (D column). null if unreadable.",
                        },
                        "damage": {
                            "type": ["integer", "null"],
                            "description": "Damage dealt (DMG column) as raw integer, no commas. null if unreadable.",
                        },
                        "healing": {
                            "type": ["integer", "null"],
                            "description": "Healing done (H column) as raw integer, no commas. null if unreadable.",
                        },
                        "mitigation": {
                            "type": ["integer", "null"],
                            "description": "Damage mitigated (MIT column) as raw integer, no commas. null if unreadable.",
                        },
                        "is_self": {
                            "type": "boolean",
                            "description": "True for the player whose Tab screen this is (highlighted row on your team).",
                        },
                        "hero": {
                            "type": ["object", "null"],
                            "description": "Hero-specific stats if visible for this player. null otherwise.",
                            "properties": {
                                "hero_name": {
                                    "type": "string",
                                    "description": "Hero name (e.g. 'Ana', 'Reinhardt'). Read from the hero stats panel, NOT the player name.",
                                },
                                "stats": {
                                    "type": "array",
                                    "description": "Hero-specific stat entries from the right panel.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {
                                                "type": "string",
                                                "description": "Stat label (e.g. 'Weapon Accuracy', 'Players Saved').",
                                            },
                                            "value": {
                                                "type": "string",
                                                "description": "Stat value as displayed (e.g. '31%', '11', '1:23').",
                                            },
                                            "is_featured": {
                                                "type": "boolean",
                                                "description": "True for the big highlighted metric in the top-right of the hero stats panel.",
                                            },
                                        },
                                        "required": ["label", "value", "is_featured"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["hero_name", "stats"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "team", "role", "player_name", "title",
                        "eliminations", "assists", "deaths",
                        "damage", "healing", "mitigation",
                        "is_self", "hero",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "not_ow2_tab", "map_name", "duration", "mode",
            "queue_type", "result", "players",
        ],
        "additionalProperties": False,
    },
}

_EXTRA_HERO_STATS_SCHEMA = {
    "type": "array",
    "description": "Hero stats from additional hero panel crops (other heroes the self-player switched to). Empty array if no extra crops provided.",
    "items": {
        "type": "object",
        "properties": {
            "hero_name": {
                "type": "string",
                "description": "Hero name read from the crop.",
            },
            "stats": {
                "type": "array",
                "description": "Hero-specific stat entries from the crop.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                        "is_featured": {"type": "boolean"},
                    },
                    "required": ["label", "value", "is_featured"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["hero_name", "stats"],
        "additionalProperties": False,
    },
}


def make_schema_with_extra_heroes() -> dict:
    """Return MATCH_SCHEMA extended with extra_hero_stats field."""
    import copy
    schema = copy.deepcopy(MATCH_SCHEMA)
    schema["schema"]["properties"]["extra_hero_stats"] = _EXTRA_HERO_STATS_SCHEMA
    schema["schema"]["required"].append("extra_hero_stats")
    return schema


SYSTEM_PROMPT = """\
You are an Overwatch 2 match analyst. You will be given a screenshot of the \
in-game Tab screen (the scoreboard that appears when a player holds Tab during \
a match). Extract all visible match data into the JSON schema provided.

Scoreboard layout (left to right per row): role icon (shield = TANK, \
bullets/crosshair = DPS, plus/cross = SUPPORT), hero portrait, ult charge %, \
player name (BattleTag, ALWAYS IN UPPERCASE) with an optional title shown \
in smaller text below it in Title Case (e.g. "Stalwart Hero", "Medic", \
"Data Broker"), up to two perk icons, then stat columns: E (eliminations), A (assists), D (deaths), DMG (damage dealt), \
H (healing done), MIT (damage mitigated/blocked).

Rules:
- Your team is the top group, enemy team is the bottom group.
- The self-player row is visually highlighted/selected on your team.
- queue_type is determined ONLY by the mode text in the top-right corner. \
If it contains "- COMPETITIVE" (e.g. "ESCORT - COMPETITIVE"), set \
queue_type to "COMPETITIVE" and strip the suffix from mode. \
If there is no "- COMPETITIVE" suffix, queue_type MUST be "QUICKPLAY". \
Do NOT infer competitive from player titles, skill level, or any other cue.
- For result: look for ATHENA subtitle text ("VICTORY"/"DEFEAT") overlaying \
the scoreboard. If absent, use the audio hint from the user message if provided. \
Otherwise use "UNKNOWN".
- DMG/H/MIT are raw integers with no commas (e.g. 8990 not "8,990").
- Hero stats panel (right side): only visible for the selected player. \
The top-right shows a big highlighted number (is_featured=true). \
Read the hero name from this panel, NOT from the player's BattleTag. \
Set hero to null for all players except the one whose hero stats panel is visible.
- If the screenshot is NOT an OW2 Tab screen, set not_ow2_tab=true and use \
empty/default values for other fields.
"""


def log_cost(model: str, input_tokens: int, output_tokens: int, cost: float,
             elapsed: float | None = None) -> None:
    """Append a line to api_costs.jsonl for tracking."""
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    }
    if elapsed is not None:
        entry["elapsed_s"] = round(elapsed, 1)
    try:
        with open(_COST_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        _logger.warning(f"Failed to write cost log: {e}")


def merge_heroes(data: dict, hero_map: dict[str, str] | None = None,
                 hero_history: dict[str, list[tuple[float, str]]] | None = None) -> dict:
    """Merge hero_history + analyzer hero + extra_hero_stats into per-player heroes arrays.

    Mutates and returns data with:
    - Each player gets a 'heroes' array
    - 'extra_hero_stats' is consumed and removed from top-level
    - 'hero' field is removed from each player (heroes[] is canonical)
    """
    from overwatchlooker.heroes import edit_distance as _edit_distance

    hero_history = hero_history or {}
    hero_map = hero_map or {}
    extra_hero_stats = data.pop("extra_hero_stats", []) or []

    # Compute match_start from hero_history timestamps
    all_times = [t for entries in hero_history.values() for t, _ in entries]
    match_start = min(all_times) if all_times else 0.0

    for p in data["players"]:
        player_name = p["player_name"]
        heroes = []

        # 1. Start with hero_history (subtitle tracking) — gives hero_name + started_at
        hist = hero_history.get(player_name, [])
        if hist:
            # Group by hero name (player may switch back to same hero)
            hero_times: dict[str, list[int]] = {}
            for t, hero_name in hist:
                start_s = max(0, int(t - match_start))
                hero_times.setdefault(hero_name, []).append(start_s)
            for hero_name, times in hero_times.items():
                heroes.append({
                    "hero_name": hero_name,
                    "started_at": times,
                    "stats": [],
                })
        elif hero_map.get(player_name):
            # Single hero from subtitle map, no switch history
            heroes.append({
                "hero_name": hero_map[player_name],
                "started_at": [0],
                "stats": [],
            })

        # 2. Merge analyzer's main hero stats (from the screenshot's hero panel)
        if p.get("is_self") and p.get("hero"):
            analyzer_hero = p["hero"]
            matched = False
            for h in heroes:
                if _edit_distance(h["hero_name"].lower(), analyzer_hero["hero_name"].lower()) <= 2:
                    h["stats"] = analyzer_hero["stats"]
                    matched = True
                    break
            if not matched:
                heroes.append({
                    "hero_name": analyzer_hero["hero_name"],
                    "started_at": [],
                    "stats": analyzer_hero["stats"],
                })

        # 3. Merge extra_hero_stats (from hero crops) — only for self player
        if p.get("is_self"):
            for extra in extra_hero_stats:
                matched = False
                for h in heroes:
                    if _edit_distance(h["hero_name"].lower(), extra["hero_name"].lower()) <= 2:
                        if not h["stats"]:  # don't overwrite main hero stats
                            h["stats"] = extra["stats"]
                        matched = True
                        break
                if not matched:
                    heroes.append({
                        "hero_name": extra["hero_name"],
                        "started_at": [],
                        "stats": extra["stats"],
                    })

        p["heroes"] = heroes
        # Remove old hero field — heroes[] is now canonical
        p.pop("hero", None)

    return data


def format_match(data: dict, hero_map: dict[str, str] | None = None,
                 hero_history: dict[str, list[tuple[float, str]]] | None = None) -> str:
    """Format a structured match dict into the display text.

    Args:
        hero_map: Optional UPPERCASE username -> hero name mapping from subtitle OCR.
        hero_history: Optional UPPERCASE username -> [(monotonic_time, hero_name), ...] for switches.
    """
    lines = []
    lines.append(f"MAP: {data['map_name']}")
    lines.append(f"TIME: {data['duration']}")
    lines.append(f"MODE: {data['mode']}")
    lines.append(f"QUEUE TYPE: {data['queue_type']}")
    lines.append(f"RESULT: {data['result']}")
    lines.append("")

    for team_key, team_label in [("ALLY", "YOUR TEAM"), ("ENEMY", "ENEMY TEAM")]:
        team_players = [p for p in data["players"] if p["team"] == team_key]
        lines.append(f"=== {team_label} ===")
        lines.append("Role | Player | E | A | D | DMG | H | MIT")
        for p in team_players:
            def _fmt(v):
                return "-" if v is None else f"{v:,}"
            name = p['player_name']
            player_hist = (hero_history or {}).get(p['player_name'], [])
            if len(player_hist) > 1:
                heroes = [h for _, h in player_hist]
                name += f" [{' > '.join(heroes)}]"
            else:
                subtitle_hero = (hero_map or {}).get(p['player_name'])
                if subtitle_hero:
                    name += f" [{subtitle_hero}]"
            if p.get('title'):
                name += f" ({p['title']})"
            lines.append(
                f"{p['role']} | {name} | "
                f"{_fmt(p['eliminations'])} | {_fmt(p['assists'])} | {_fmt(p['deaths'])} | "
                f"{_fmt(p['damage'])} | {_fmt(p['healing'])} | {_fmt(p['mitigation'])}"
            )
        lines.append("")

    # Hero stats from merged heroes[] arrays
    hero_entries = []
    for p in data["players"]:
        for h in p.get("heroes", []):
            if h.get("stats"):
                parts = []
                featured = [s for s in h["stats"] if s.get("is_featured")]
                regular = [s for s in h["stats"] if not s.get("is_featured")]
                for s in featured:
                    parts.append(f"{s['label']}: {s['value']}")
                for s in regular:
                    parts.append(f"{s['label']}: {s['value']}")
                hero_entries.append(f"{h['hero_name']} - {'; '.join(parts)}")

    if hero_entries:
        lines.append("HERO STATS:")
        lines.extend(hero_entries)

    # Hero switches section (players who switched heroes during the match)
    if hero_history:
        all_times = [t for entries in hero_history.values() for t, _ in entries]
        if all_times:
            match_start = min(all_times)
            switch_lines = []
            player_names = {p['player_name'] for p in data['players']}
            for username in sorted(hero_history):
                if username not in player_names:
                    continue
                entries = hero_history[username]
                if len(entries) <= 1:
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
                switch_lines.append(f"{username}: {', '.join(parts)}")
            if switch_lines:
                lines.append("")
                lines.append("HERO SWITCHES:")
                lines.extend(switch_lines)

    return "\n".join(lines)
