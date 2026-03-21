"""MCP client for submitting matches via the official SDK."""

import asyncio
import base64
import datetime
import json
import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

from overwatchlooker.config import MCP_SOURCE, MCP_URL

_logger = logging.getLogger("overwatchlooker")

def submit_match(
    data: dict,
    png_bytes: bytes | None = None,
    is_backfill: bool = False,
) -> dict:
    """Submit a match to the MCP server. Returns dict with match_id if available."""
    if not MCP_URL:
        raise RuntimeError("MCP_URL not set in .env")

    return asyncio.run(_submit_match_async(data, png_bytes, is_backfill))


async def _submit_match_async(
    data: dict,
    png_bytes: bytes | None,
    is_backfill: bool,
) -> dict:
    _logger.info(f"MCP: submitting match ({data['map_name']}, {data['mode']}, {data['queue_type']})")

    args: dict = {
        "map_name": data["map_name"],
        "duration": data["duration"],
        "mode": data["mode"],
        "queue_type": data["queue_type"],
        "result": data["result"],
        "players": data["players"],
        "source": MCP_SOURCE,
        "played_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_backfill": is_backfill,
    }

    # Pass through optional rank fields
    if data.get("rank_min"):
        args["rank_min"] = data["rank_min"]
    if data.get("rank_max"):
        args["rank_max"] = data["rank_max"]
    if data.get("is_wide_match"):
        args["is_wide_match"] = True
    if data.get("banned_heroes"):
        args["banned_heroes"] = data["banned_heroes"]

    if png_bytes:
        args["screenshot_uploads"] = [{
            "data": base64.standard_b64encode(png_bytes).decode("utf-8"),
            "filename": "screenshot.png",
        }]

    async with streamablehttp_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("submit_match", args)

    if result.isError:
        text = " ".join(
            b.text for b in result.content if isinstance(b, TextContent)
        )
        _logger.error(f"MCP error: {text}")
        raise RuntimeError(f"MCP error: {text}")

    _logger.info("MCP: match submitted successfully")

    match_id = _extract_match_id(result)
    return {"match_id": match_id} if match_id else {}


def _extract_match_id(result) -> str | None:
    """Extract match ID from CallToolResult via structured content or JSON text."""
    if result.structuredContent:
        for key in ("id", "match_id"):
            if key in result.structuredContent:
                return str(result.structuredContent[key])

    for block in result.content:
        if not isinstance(block, TextContent):
            continue
        try:
            data = json.loads(block.text)
            if isinstance(data, dict):
                for key in ("id", "match_id"):
                    if key in data:
                        return str(data[key])
        except (json.JSONDecodeError, TypeError):
            pass
    return None
