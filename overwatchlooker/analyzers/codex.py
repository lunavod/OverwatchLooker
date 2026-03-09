"""Codex (ChatGPT) vision-based scoreboard analyzer for Overwatch 2 Tab screens."""

import base64
import io
import json
import logging

import codex_open_client
from PIL import Image

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
import litellm  # noqa: E402
from litellm import cost_per_token  # noqa: E402

litellm.suppress_debug_info = True

from overwatchlooker.analyzers.common import MATCH_SCHEMA, SYSTEM_PROMPT, log_cost
from overwatchlooker.config import CODEX_MODEL, CODEX_REASONING, OVERWATCH_USERNAME

_logger = logging.getLogger("overwatchlooker")

# Scoreboard player names region (fraction of image: x1, y1, x2, y2)
_NAMES_REGION = (0.15, 0.15, 0.45, 0.85)


def _crop_names(png_bytes: bytes) -> bytes:
    """Crop the player names area from the scoreboard for better OCR."""
    img = Image.open(io.BytesIO(png_bytes))
    w, h = img.size
    x1, y1, x2, y2 = _NAMES_REGION
    crop = img.crop((int(w * x1), int(h * y1), int(w * x2), int(h * y2)))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def _to_data_url(png_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def analyze_screenshot(png_bytes: bytes, audio_result: str | None = None) -> dict:
    """Send screenshot to Codex/ChatGPT and return structured match data."""
    from overwatchlooker.display import print_status

    names_crop = _crop_names(png_bytes)
    from overwatchlooker.screenshot import resize_for_analyzer
    png_bytes = resize_for_analyzer(png_bytes, "codex")
    print_status(f"Sending to Codex ({CODEX_MODEL})...")

    client = codex_open_client.CodexClient()

    user_text = (
        "Analyze this Overwatch 2 Tab screen screenshot. "
        "The second image is a zoomed crop of the player names area for better readability."
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

    response = client.responses.create(
        model=CODEX_MODEL,
        instructions=SYSTEM_PROMPT,
        input=[
            codex_open_client.InputMessage(
                role="user",
                content=[
                    codex_open_client.InputImage(image_url=_to_data_url(png_bytes), detail="high"),
                    codex_open_client.InputImage(image_url=_to_data_url(names_crop), detail="high"),
                    codex_open_client.InputText(text=user_text),
                ],
            )
        ],
        text=codex_open_client.TextConfig(
            format=codex_open_client.ResponseFormatJsonSchema(
                name=MATCH_SCHEMA["name"],
                schema=MATCH_SCHEMA["schema"],
                strict=True,
            )
        ),
        reasoning=codex_open_client.Reasoning(effort=CODEX_REASONING) if CODEX_REASONING else None,
    )

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
                         f"est. API cost: ${total_cost:.4f}")
            log_cost(CODEX_MODEL, usage.input_tokens, usage.output_tokens, total_cost)
        except Exception:
            print_status(f"Tokens -- input: {usage.input_tokens}, output: {usage.output_tokens}")
            log_cost(CODEX_MODEL, usage.input_tokens, usage.output_tokens, 0.0)

    return json.loads(response.output_text)
