import base64

from openai import OpenAI

from overwatchlooker.config import (
    MAX_TOKENS,
    OLLAMA_BASE_URL,
    VISION_MODEL,
)

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
- For RESULT: only write VICTORY or DEFEAT if you can clearly see the ATHENA \
subtitle text on screen. If there is no such subtitle, write "UNKNOWN".
- If the screenshot is NOT an Overwatch 2 Tab screen, respond with exactly: \
NOT_OW2_TAB: This does not appear to be an Overwatch 2 scoreboard screenshot.
"""


def analyze_screenshot(png_bytes: bytes) -> str:
    """Send screenshot to local Ollama vision model and return the analysis text."""
    image_data = base64.standard_b64encode(png_bytes).decode("utf-8")

    from overwatchlooker.display import print_status

    model = VISION_MODEL
    print_status(f"Image payload: {len(image_data)} chars base64")
    print_status(f"Backend: Ollama, Model: {model}, max_tokens: {MAX_TOKENS}")

    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    print_status("Opening stream...")
    stream = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze this Overwatch 2 Tab screen screenshot. Extract all visible match data.",
                    },
                ],
            },
        ],
        stream=True,
    )

    result_text = ""
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice and choice.delta and choice.delta.content:
            result_text += choice.delta.content

        if choice and choice.finish_reason:
            print_status(f"Stream done (finish_reason: {choice.finish_reason})")

    if hasattr(stream, "usage") and stream.usage:
        usage = stream.usage
        print_status(
            f"Tokens -- input: {usage.prompt_tokens}, output: {usage.completion_tokens}"
        )

    return result_text
