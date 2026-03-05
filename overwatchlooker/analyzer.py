"""Claude Vision-based scoreboard analyzer for Overwatch 2 Tab screens."""

import base64
import json
import logging
from pathlib import Path

import anthropic

# Suppress litellm debug spam before import
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
import litellm  # noqa: E402
from litellm import cost_per_token  # noqa: E402

litellm.suppress_debug_info = True

from overwatchlooker.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_TOKENS, OVERWATCH_USERNAME

_logger = logging.getLogger("overwatchlooker")

_COST_LOG = Path(__file__).parent.parent / "api_costs.jsonl"

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
                            "description": "Player BattleTag exactly as shown.",
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
                        "team", "role", "player_name",
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


def _log_cost(model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
    """Append a line to api_costs.jsonl for tracking."""
    import datetime
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    }
    try:
        with open(_COST_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        _logger.warning(f"Failed to write cost log: {e}")

SYSTEM_PROMPT = """\
You are an Overwatch 2 match analyst. You will be given a screenshot of the \
in-game Tab screen (the scoreboard that appears when a player holds Tab during \
a match). Extract all visible match data into the JSON schema provided.

Scoreboard layout (left to right per row): role icon (shield = TANK, \
bullets/crosshair = DPS, plus/cross = SUPPORT), hero portrait, ult charge %, \
player name (BattleTag), up to two perk icons, then stat columns: \
E (eliminations), A (assists), D (deaths), DMG (damage dealt), \
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


def analyze_screenshot(png_bytes: bytes, audio_result: str | None = None) -> dict:
    """Send screenshot to Claude Vision and return structured match data."""
    from overwatchlooker.display import print_status

    image_data = base64.standard_b64encode(png_bytes).decode("utf-8")
    print_status(f"Sending to Claude ({ANTHROPIC_MODEL})...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_text = "Analyze this Overwatch 2 Tab screen screenshot."
    if OVERWATCH_USERNAME:
        user_text += (
            f"\n\nThe player's username is \"{OVERWATCH_USERNAME}\". "
            "This player MUST always be present in the output and MUST have is_self=true. "
            "Do NOT include player titles (text shown below the username) in the player_name field."
        )
    if audio_result:
        user_text += (
            f"\n\nNote: audio detection identified this as a {audio_result}. "
            "Use this as fallback if you cannot read the result text on screen."
        )

    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": MATCH_SCHEMA["schema"],
            },
        },
    )

    usage = message.usage
    inp_cost, out_cost = cost_per_token(
        model=ANTHROPIC_MODEL,
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
    )
    total_cost = inp_cost + out_cost
    print_status(f"Tokens -- input: {usage.input_tokens}, output: {usage.output_tokens}, "
                 f"cost: ${total_cost:.4f}")
    _log_cost(ANTHROPIC_MODEL, usage.input_tokens, usage.output_tokens, total_cost)

    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)


def format_match(data: dict) -> str:
    """Format a structured match dict into the display text."""
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
            lines.append(
                f"{p['role']} | {p['player_name']} | "
                f"{_fmt(p['eliminations'])} | {_fmt(p['assists'])} | {_fmt(p['deaths'])} | "
                f"{_fmt(p['damage'])} | {_fmt(p['healing'])} | {_fmt(p['mitigation'])}"
            )
        lines.append("")

    # Hero stats
    hero_entries = []
    for p in data["players"]:
        hero = p.get("hero")
        if hero:
            parts = []
            featured = [s for s in hero["stats"] if s.get("is_featured")]
            regular = [s for s in hero["stats"] if not s.get("is_featured")]
            for s in featured:
                parts.append(f"{s['label']}: {s['value']}")
            for s in regular:
                parts.append(f"{s['label']}: {s['value']}")
            hero_entries.append(f"{hero['hero_name']} - {'; '.join(parts)}")

    if hero_entries:
        lines.append("HERO STATS:")
        lines.extend(hero_entries)

    return "\n".join(lines)
