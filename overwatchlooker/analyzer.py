import base64

import anthropic

from overwatchlooker.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    AWS_KEY,
    AWS_REGION,
    AWS_SECRET,
    BEDROCK_MODEL,
    MAX_TOKENS,
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


def _use_bedrock() -> bool:
    return bool(AWS_KEY and AWS_SECRET)


def analyze_screenshot(png_bytes: bytes) -> str:
    """Send screenshot to Claude Vision and return the analysis text."""
    image_data = base64.standard_b64encode(png_bytes).decode("utf-8")

    from overwatchlooker.display import print_status

    bedrock = _use_bedrock()
    backend = "Bedrock" if bedrock else "Anthropic"
    model = BEDROCK_MODEL if bedrock else ANTHROPIC_MODEL
    print_status(f"Image payload: {len(image_data)} chars base64")
    print_status(f"Backend: {backend}, Model: {model}, max_tokens: {MAX_TOKENS}")

    if bedrock:
        client = anthropic.AnthropicBedrock(
            aws_access_key=AWS_KEY,
            aws_secret_key=AWS_SECRET,
            aws_region=AWS_REGION,
        )
    else:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print_status("Opening stream...")
    with client.messages.stream(
        model=model,
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
                    {
                        "type": "text",
                        "text": "Analyze this Overwatch 2 Tab screen screenshot. Extract all visible match data.",
                    },
                ],
            }
        ],
    ) as stream:
        current_block = None
        for event in stream:
            if event.type == "content_block_start":
                block_type = event.content_block.type
                if block_type == "thinking":
                    current_block = "thinking"
                    print_status("Thinking started...")
                    print("[THINKING] ", end="", flush=True)
                elif block_type == "text":
                    current_block = "text"
                    print_status("Generating response...")

            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(event.delta.thinking, end="", flush=True)
                elif event.delta.type == "text_delta":
                    pass  # don't print response text live, we print it formatted later

            elif event.type == "content_block_stop":
                if current_block == "thinking":
                    print()  # newline after thinking output
                    print_status("Thinking complete.")
                current_block = None

            elif event.type == "message_delta":
                reason = getattr(event.delta, "stop_reason", None)
                if reason:
                    print_status(f"Stream done (stop_reason: {reason})")

        message = stream.get_final_message()

    usage = message.usage
    print_status(
        f"Tokens -- input: {usage.input_tokens}, output: {usage.output_tokens}"
    )

    return next(block.text for block in message.content if block.type == "text")
