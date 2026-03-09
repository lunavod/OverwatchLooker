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

from overwatchlooker.analyzers.common import MATCH_SCHEMA, SYSTEM_PROMPT, log_cost
from overwatchlooker.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_TOKENS, OVERWATCH_USERNAME

_logger = logging.getLogger("overwatchlooker")


def analyze_screenshot(png_bytes: bytes, audio_result: str | None = None) -> dict:
    """Send screenshot to Claude Vision and return structured match data."""
    from overwatchlooker.display import print_status

    from overwatchlooker.screenshot import resize_for_analyzer
    png_bytes = resize_for_analyzer(png_bytes, "anthropic")
    image_data = base64.standard_b64encode(png_bytes).decode("utf-8")
    print_status(f"Sending to Claude ({ANTHROPIC_MODEL})...")
    t0 = time.monotonic()

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

    elapsed = time.monotonic() - t0
    usage = message.usage
    inp_cost, out_cost = cost_per_token(
        model=ANTHROPIC_MODEL,
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
    )
    total_cost = inp_cost + out_cost
    print_status(f"Tokens -- input: {usage.input_tokens}, output: {usage.output_tokens}, "
                 f"cost: ${total_cost:.4f}, time: {elapsed:.1f}s")
    log_cost(ANTHROPIC_MODEL, usage.input_tokens, usage.output_tokens, total_cost, elapsed)

    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)
