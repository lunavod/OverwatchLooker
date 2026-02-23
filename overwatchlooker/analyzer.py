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

from overwatchlooker.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_TOKENS

_logger = logging.getLogger("overwatchlooker")

_COST_LOG = Path(__file__).parent.parent / "api_costs.jsonl"


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
a match).

Extract and report the following information in a structured format:

1. MAP: The map name (shown at the top of the screen).
2. MATCH TIME: The elapsed match time or round timer visible on screen.
3. GAME MODE: The game mode if identifiable (Push, Control, Escort, Hybrid, Clash, Flashpoint).
4. MATCH RESULT: Check for ATHENA subtitle text on screen announcing \
"VICTORY" or "DEFEAT". This appears as large centered text overlaying the \
scoreboard at the end of a round or match.
5. YOUR TEAM and ENEMY TEAM scoreboard stats:
   Each row on the scoreboard has the following layout from left to right: \
role icon (shield = TANK, bullets/crosshair = DPS, plus/cross = SUPPORT), \
hero portrait, player ult charge percentage, player name (BattleTag), \
up to two white circular icons representing chosen perks, then the stats \
columns: E (eliminations), A (assists), D (deaths), DMG (damage dealt), \
H (healing done), MIT (damage mitigated/blocked).
   For EACH player, report:
   - Player name (BattleTag)
   - Role (TANK, DPS, SUPPORT) -- identified from the role icon
   - E / A / D / DMG / H / MIT
   Do NOT attempt to identify heroes from portraits. Ignore the hero portrait column.

Format your response EXACTLY as follows (do not deviate from this structure):

MAP: <map name>
TIME: <match time>
MODE: <game mode>
RESULT: <VICTORY, DEFEAT, or UNKNOWN>

=== YOUR TEAM ===
Role | Player | E | A | D | DMG | H | MIT
<row for each player>

=== ENEMY TEAM ===
Role | Player | E | A | D | DMG | H | MIT
<row for each player>

HERO STATS:
<For each row with visible hero-specific stats, list them as:>
<HeroName> - <top-right metric>; <stat1>, <stat2>, ...
IMPORTANT: Use the HERO NAME (e.g. "Ana", "Reinhardt"), NOT the player's \
username/BattleTag. The hero name can be read from the hero-specific stats \
panel on the right side of the screen, or from the hero portrait tooltip.
In the top right corner of the right panel, there is a highlighted metric \
displayed as a BIG WHITE NUMBER with smaller grey label text underneath it. \
You MUST include this metric first, formatted as "Label: Value", before \
listing the other hero-specific stats.

Rules:
- If a stat is not visible or not applicable, write "-".
- Role is one of: TANK, DPS, SUPPORT.
- Keep player names exactly as shown (including any special characters).
- For RESULT: write VICTORY or DEFEAT if you can clearly see the ATHENA \
subtitle text on screen. If there is no such subtitle but the user message \
includes an audio detection result, use that. Otherwise write "UNKNOWN".
- If the screenshot is NOT an Overwatch 2 Tab screen, respond with exactly: \
NOT_OW2_TAB: This does not appear to be an Overwatch 2 scoreboard screenshot.
"""


def analyze_screenshot(png_bytes: bytes, audio_result: str | None = None) -> str:
    """Send screenshot to Claude Vision and return the analysis text."""
    from overwatchlooker.display import print_status

    image_data = base64.standard_b64encode(png_bytes).decode("utf-8")
    print_status(f"Sending to Claude ({ANTHROPIC_MODEL})...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_text = "Analyze this Overwatch 2 Tab screen screenshot. Extract all visible match data."
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

    return next(block.text for block in message.content if block.type == "text")
