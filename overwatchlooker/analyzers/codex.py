"""Codex (ChatGPT) vision-based scoreboard analyzer for Overwatch 2 Tab screens."""

import base64
import json
import logging
import time

import codex_open_client

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
import litellm  # noqa: E402
from litellm import cost_per_token  # noqa: E402

litellm.suppress_debug_info = True

from overwatchlooker.analyzers.common import (  # noqa: E402
    MATCH_SCHEMA, NAMES_REGION, RANK_REGION, SYSTEM_PROMPT, crop_region, log_cost,
)
from overwatchlooker.config import CODEX_MODEL, CODEX_REASONING, OVERWATCH_USERNAME  # noqa: E402

_logger = logging.getLogger("overwatchlooker")


def _to_data_url(png_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def analyze_screenshot(png_bytes: bytes, audio_result: str | None = None,
                       hero_crops: dict[str, bytes] | None = None) -> dict:
    """Send screenshot to Codex/ChatGPT and return structured match data."""
    from overwatchlooker.display import print_status

    names_crop = crop_region(png_bytes, NAMES_REGION)
    rank_crop = crop_region(png_bytes, RANK_REGION)
    from overwatchlooker.screenshot import resize_for_analyzer
    png_bytes = resize_for_analyzer(png_bytes, "codex")
    crop_count = len(hero_crops) if hero_crops else 0
    print_status(f"Sending to Codex ({CODEX_MODEL}), {crop_count} extra hero crops...")
    t0 = time.monotonic()

    client = codex_open_client.CodexClient()

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

    content: list[codex_open_client.InputText | codex_open_client.InputImage] = [
        codex_open_client.InputImage(image_url=_to_data_url(png_bytes), detail="high"),
        codex_open_client.InputImage(image_url=_to_data_url(names_crop), detail="high"),
        codex_open_client.InputImage(image_url=_to_data_url(rank_crop), detail="high"),
    ]
    if hero_crops:
        for hero_name, crop_bytes in hero_crops.items():
            content.append(codex_open_client.InputImage(
                image_url=_to_data_url(crop_bytes), detail="high"))
    content.append(codex_open_client.InputText(text=user_text))

    # Build schema — add extra_hero_stats if we have hero crops
    schema = MATCH_SCHEMA
    if hero_crops:
        from overwatchlooker.analyzers.common import make_schema_with_extra_heroes
        schema = make_schema_with_extra_heroes()

    from typing import Any, cast
    schema_name = cast(str, schema["name"])
    schema_body = cast(dict[str, Any], schema["schema"])

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
            print_status(f"Tokens -- input: {usage.input_tokens}, output: {usage.output_tokens}, "
                         f"est. API cost: ${total_cost:.4f}, time: {elapsed:.1f}s")
            log_cost(CODEX_MODEL, usage.input_tokens, usage.output_tokens, total_cost, elapsed)
        except Exception:
            print_status(f"Tokens -- input: {usage.input_tokens}, output: {usage.output_tokens}, "
                         f"time: {elapsed:.1f}s")
            log_cost(CODEX_MODEL, usage.input_tokens, usage.output_tokens, 0.0, elapsed)

    return json.loads(response.output_text)
