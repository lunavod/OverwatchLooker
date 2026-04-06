"""Codex (ChatGPT) vision-based Tab screen analyzer for fallback mode.

When Overwolf GEP is unavailable, this module sends the Tab screenshot to
a ChatGPT vision model and extracts scoreboard data (map, mode, duration,
players with stats) via structured output.
"""

import base64
import datetime
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

from PIL import Image

from overwatchlooker.config import CODEX_MODEL, CODEX_REASONING, OVERWATCH_USERNAME

_logger = logging.getLogger("overwatchlooker")

_COST_LOG = Path(__file__).parent.parent / "api_costs.jsonl"

# Crop regions (fraction of image: x1, y1, x2, y2)
_NAMES_REGION = (0.15, 0.15, 0.45, 0.85)
_RANK_REGION = (0.75, 0.0, 1.0, 0.13)


# ---------------------------------------------------------------------------
# JSON schema for structured output
# ---------------------------------------------------------------------------

MATCH_SCHEMA: dict[str, Any] = {
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
            "players": {
                "type": "array",
                "description": "All 10 players from both teams.",
                "items": {
                    "type": "object",
                    "properties": {
                        "team": {
                            "type": "string",
                            "enum": ["ALLY", "ENEMY"],
                        },
                        "role": {
                            "type": "string",
                            "enum": ["TANK", "DPS", "SUPPORT"],
                        },
                        "player_name": {
                            "type": "string",
                            "description": "Player BattleTag in UPPERCASE exactly as shown.",
                        },
                        "title": {
                            "type": ["string", "null"],
                            "description": "Title below player name in Title Case, or null.",
                        },
                        "hero_name": {
                            "type": ["string", "null"],
                            "description": "Hero name from the hero stats panel (right side). null for players without visible panel.",
                        },
                        "eliminations": {"type": ["integer", "null"]},
                        "assists": {"type": ["integer", "null"]},
                        "deaths": {"type": ["integer", "null"]},
                        "damage": {
                            "type": ["integer", "null"],
                            "description": "Raw integer, no commas.",
                        },
                        "healing": {
                            "type": ["integer", "null"],
                            "description": "Raw integer, no commas.",
                        },
                        "mitigation": {
                            "type": ["integer", "null"],
                            "description": "Raw integer, no commas.",
                        },
                        "is_self": {
                            "type": "boolean",
                            "description": "True for the highlighted/selected player row.",
                        },
                    },
                    "required": [
                        "team", "role", "player_name", "title", "hero_name",
                        "eliminations", "assists", "deaths",
                        "damage", "healing", "mitigation", "is_self",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["not_ow2_tab", "map_name", "duration", "mode",
                      "queue_type", "players"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """\
You are an Overwatch 2 match analyst. You will be given a screenshot of the \
in-game Tab screen (the scoreboard that appears when a player holds Tab during \
a match). Extract all visible match data into the JSON schema provided.

Scoreboard layout (left to right per row): role icon (shield = TANK, \
bullets/crosshair = DPS, plus/cross = SUPPORT), hero portrait, ult charge %, \
player name (BattleTag, ALWAYS IN UPPERCASE) with an optional title shown \
in smaller text below it in Title Case (e.g. "Stalwart Hero", "Medic", \
"Data Broker"), up to two perk icons, then stat columns: E (eliminations), \
A (assists), D (deaths), DMG (damage dealt), H (healing done), \
MIT (damage mitigated/blocked).

Rules:
- Your team is the top group, enemy team is the bottom group.
- The self-player row is visually highlighted/selected on your team.
- queue_type is determined ONLY by the mode text in the top-right corner. \
If it contains "- COMPETITIVE" (e.g. "ESCORT - COMPETITIVE"), set \
queue_type to "COMPETITIVE" and strip the suffix from mode. \
If there is no "- COMPETITIVE" suffix, queue_type MUST be "QUICKPLAY".
- DMG/H/MIT are raw integers with no commas (e.g. 8990 not "8,990").
- Hero stats panel (right side): only visible for the selected player. \
Read the hero name from this panel, NOT from the player's BattleTag. \
Set hero_name to null for all players except the one whose hero stats \
panel is visible.
- If the screenshot is NOT an OW2 Tab screen, set not_ow2_tab=true and use \
empty/default values for other fields.
"""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _crop_region(png_bytes: bytes, region: tuple[float, float, float, float]) -> bytes:
    """Crop a region from the image. Region is (x1, y1, x2, y2) as fractions."""
    img = Image.open(io.BytesIO(png_bytes))
    w, h = img.size
    x1, y1, x2, y2 = region
    crop = img.crop((int(w * x1), int(h * y1), int(w * x2), int(h * y2)))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def _to_data_url(png_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Cost logging
# ---------------------------------------------------------------------------

def _log_cost(model: str, input_tokens: int, output_tokens: int,
              cost: float, elapsed: float | None = None) -> None:
    entry: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# Main API call
# ---------------------------------------------------------------------------

def analyze_tab_screenshot(png_bytes: bytes) -> dict[str, Any]:
    """Send Tab screenshot to Codex and return structured match data.

    Returns the parsed JSON dict matching MATCH_SCHEMA, or raises on failure.
    """
    import codex_open_client
    from litellm import cost_per_token  # type: ignore[import-untyped]
    from overwatchlooker.display import print_status

    names_crop = _crop_region(png_bytes, _NAMES_REGION)
    rank_crop = _crop_region(png_bytes, _RANK_REGION)

    print_status(f"Fallback: sending tab to {CODEX_MODEL}...")
    t0 = time.monotonic()

    client = codex_open_client.CodexClient()

    user_text = (
        "Analyze this Overwatch 2 Tab screen screenshot. "
        "The second image is a zoomed crop of the player names area. "
        "The third image is a zoomed crop of the top-right corner "
        "showing the mode, rank range, and match time."
    )
    if OVERWATCH_USERNAME:
        user_text += (
            f"\n\nThe player's username is \"{OVERWATCH_USERNAME}\". "
            "This player MUST always be present in the output and MUST have is_self=true. "
            "Do NOT include player titles in the player_name field."
        )

    content: list[codex_open_client.InputText | codex_open_client.InputImage] = [
        codex_open_client.InputImage(image_url=_to_data_url(png_bytes), detail="high"),
        codex_open_client.InputImage(image_url=_to_data_url(names_crop), detail="high"),
        codex_open_client.InputImage(image_url=_to_data_url(rank_crop), detail="high"),
        codex_open_client.InputText(text=user_text),
    ]

    schema_name = cast(str, MATCH_SCHEMA["name"])
    schema_body = cast(dict[str, Any], MATCH_SCHEMA["schema"])

    response = client.responses.create(
        model=CODEX_MODEL,
        instructions=SYSTEM_PROMPT,
        input=[
            codex_open_client.InputMessage(role="user", content=content)
        ],
        text=codex_open_client.TextConfig(
            format=codex_open_client.ResponseFormatJsonSchema(
                name=schema_name,
                schema=schema_body,
                strict=True,
            )
        ),
        reasoning=codex_open_client.Reasoning(
            effort=cast(Any, CODEX_REASONING)
        ) if CODEX_REASONING else None,
    )

    elapsed = time.monotonic() - t0
    usage = response.usage
    if usage:
        try:
            inp_cost, out_cost = cost_per_token(
                model=CODEX_MODEL,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )
            total_cost = inp_cost + out_cost
            print_status(f"Fallback LLM: {usage.input_tokens}in/{usage.output_tokens}out, "
                         f"${total_cost:.4f}, {elapsed:.1f}s")
            _log_cost(CODEX_MODEL, usage.input_tokens, usage.output_tokens,
                      total_cost, elapsed)
        except Exception:
            print_status(f"Fallback LLM: {usage.input_tokens}in/{usage.output_tokens}out, "
                         f"{elapsed:.1f}s")
            _log_cost(CODEX_MODEL, usage.input_tokens, usage.output_tokens, 0.0, elapsed)

    return json.loads(response.output_text)
