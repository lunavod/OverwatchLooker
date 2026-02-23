"""Send analysis results to Telegram via Bot API."""

import json
import logging
import urllib.request
import urllib.error

from overwatchlooker.config import TELEGRAM_CHANNEL, TELEGRAM_TOKEN

_logger = logging.getLogger("overwatchlooker")


def send_message(text: str) -> bool:
    """Send a text message to the configured Telegram channel. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHANNEL:
        _logger.error("TELEGRAM_TOKEN or TELEGRAM_CHANNEL not set in .env")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": TELEGRAM_CHANNEL,
        "text": text,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                _logger.info("Telegram message sent.")
                return True
            _logger.warning(f"Telegram API returned status {resp.status}")
            return False
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        _logger.error(f"Telegram API error {e.code}: {body}")
        return False
    except Exception as e:
        _logger.error(f"Failed to send Telegram message: {e}")
        return False
