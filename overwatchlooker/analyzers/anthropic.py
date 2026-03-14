"""Claude Vision-based scoreboard analyzer for Overwatch 2 Tab screens."""

import base64
import json
import logging
import time

import anthropic

# Suppress litellm debug spam before import
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
import litellm  # noqa: E402
from litellm import cost_per_token  # noqa: E402

litellm.suppress_debug_info = True

from overwatchlooker.analyzers.common import (
    MATCH_SCHEMA, NAMES_REGION, RANK_REGION, SYSTEM_PROMPT, crop_region, log_cost,
)
from overwatchlooker.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_TOKENS, OVERWATCH_USERNAME, SONNET_RANK_TIERS

_logger = logging.getLogger("overwatchlooker")


def _b64_image_block(png_bytes: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode("utf-8"),
        },
    }


def analyze_screenshot(png_bytes: bytes, audio_result: str | None = None,
                       hero_crops: dict[str, bytes] | None = None) -> dict:
    """Send screenshot to Claude Vision and return structured match data."""
    from overwatchlooker.display import print_status

    names_crop = crop_region(png_bytes, NAMES_REGION)
    rank_crop = crop_region(png_bytes, RANK_REGION)
    from overwatchlooker.screenshot import resize_for_analyzer
    png_bytes = resize_for_analyzer(png_bytes, "anthropic")
    crop_count = len(hero_crops) if hero_crops else 0
    print_status(f"Sending to Claude ({ANTHROPIC_MODEL}), {crop_count} extra hero crops...")
    t0 = time.monotonic()

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_text = (
        "Analyze this Overwatch 2 Tab screen screenshot. "
        "The second image is a zoomed crop of the player names area for better readability. "
        "The third image is a zoomed crop of the top-right corner showing the mode, rank range, and match time."
    )
    if hero_crops:
        user_text += (
            f"\n\nAdditional images show hero stat panels for {len(hero_crops)} other heroes "
            "the player switched to during this match. For each, read the hero name and stats. "
            "Return them in the 'extra_hero_stats' array at the top level."
        )
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

    # Sonnet can't reliably identify rank tier icons — only extract is_wide
    if "opus" not in ANTHROPIC_MODEL and not SONNET_RANK_TIERS:
        user_text += (
            "\n\nFor rank_range: do NOT attempt to identify rank tier names from the icons. "
            "Set min_rank and max_rank to empty strings. Only identify whether "
            "\"WIDE MATCH\" is shown (is_wide)."
        )

    content: list[dict] = [
        _b64_image_block(png_bytes),
        _b64_image_block(names_crop),
        _b64_image_block(rank_crop),
    ]
    if hero_crops:
        for hero_name, crop_bytes in hero_crops.items():
            content.append(_b64_image_block(crop_bytes))
    content.append({"type": "text", "text": user_text})

    # Build schema — add extra_hero_stats if we have hero crops
    schema = MATCH_SCHEMA
    if hero_crops:
        from overwatchlooker.analyzers.common import make_schema_with_extra_heroes
        schema = make_schema_with_extra_heroes()

    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": content},
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema["schema"],
            },
        },
    )

    elapsed = time.monotonic() - t0
    usage = message.usage
    try:
        inp_cost, out_cost = cost_per_token(
            model=ANTHROPIC_MODEL,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
        )
        total_cost = inp_cost + out_cost
        print_status(f"Tokens -- input: {usage.input_tokens}, output: {usage.output_tokens}, "
                     f"cost: ${total_cost:.4f}, time: {elapsed:.1f}s")
        log_cost(ANTHROPIC_MODEL, usage.input_tokens, usage.output_tokens, total_cost, elapsed)
    except Exception:
        print_status(f"Tokens -- input: {usage.input_tokens}, output: {usage.output_tokens}, "
                     f"time: {elapsed:.1f}s")
        log_cost(ANTHROPIC_MODEL, usage.input_tokens, usage.output_tokens, 0.0, elapsed)

    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)
