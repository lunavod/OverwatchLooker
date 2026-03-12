"""Lightweight MCP client for submitting matches via Streamable HTTP."""

import base64
import datetime
import logging

import httpx

from overwatchlooker.config import MCP_SOURCE, MCP_URL

_logger = logging.getLogger("overwatchlooker")


def submit_match(
    data: dict,
    png_bytes: bytes | None = None,
    is_backfill: bool = False,
) -> dict:
    """Submit a match to the MCP server. Returns the tool result."""
    if not MCP_URL:
        raise RuntimeError("MCP_URL not set in .env")

    _logger.info(f"MCP: submitting match ({data['map_name']}, {data['mode']}, {data['queue_type']})")

    players = data["players"]

    # Build arguments matching the submit_match tool schema
    args = {
        "map_name": data["map_name"],
        "duration": data["duration"],
        "mode": data["mode"],
        "queue_type": data["queue_type"],
        "result": data["result"],
        "players": players,
        "source": MCP_SOURCE,
        "played_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_backfill": is_backfill,
    }

    if png_bytes:
        args["screenshot_uploads"] = [{
            "data": base64.standard_b64encode(png_bytes).decode("utf-8"),
            "filename": "screenshot.png",
        }]

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "submit_match",
            "arguments": args,
        },
    }

    mcp_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    # Initialize session, then call tool
    with httpx.Client(timeout=30, headers=mcp_headers) as client:
        # Send initialize
        init_resp = client.post(MCP_URL, json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "overwatchlooker", "version": "1.0.0"},
            },
        })
        init_resp.raise_for_status()
        session_id = init_resp.headers.get("mcp-session-id")

        headers = {}
        if session_id:
            headers["mcp-session-id"] = session_id

        # Send initialized notification
        client.post(MCP_URL, json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }, headers=headers)

        # Call tool
        resp = client.post(MCP_URL, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()

    if "error" in result:
        _logger.error(f"MCP error: {result['error']}")
        raise RuntimeError(f"MCP error: {result['error']}")

    _logger.info("MCP: match submitted successfully")
    return result.get("result", {})
