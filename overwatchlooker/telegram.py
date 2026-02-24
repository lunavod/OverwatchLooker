"""Send analysis results to Telegram as a user account via Telethon (MTProto)."""

import asyncio
import logging
from pathlib import Path

from telethon import TelegramClient

from overwatchlooker.config import TELEGRAM_API_HASH, TELEGRAM_API_ID, TELEGRAM_CHANNEL

_logger = logging.getLogger("overwatchlooker")

_SESSION_PATH = str(Path(__file__).parent.parent / "telegram_session")


async def _send(text: str) -> None:
    async with TelegramClient(_SESSION_PATH, int(TELEGRAM_API_ID), TELEGRAM_API_HASH) as client:
        await client.send_message(int(TELEGRAM_CHANNEL), text)


def send_message(text: str) -> bool:
    """Send a text message to the configured Telegram channel. Returns True on success."""
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH or not TELEGRAM_CHANNEL:
        _logger.error("TELEGRAM_API_ID, TELEGRAM_API_HASH, or TELEGRAM_CHANNEL not set in .env")
        return False

    try:
        asyncio.run(_send(text))
        _logger.info("Telegram message sent.")
        return True
    except Exception as e:
        _logger.error(f"Failed to send Telegram message: {e}")
        return False
