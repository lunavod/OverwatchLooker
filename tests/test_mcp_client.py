"""Tests for MCP client: argument building, match_id extraction."""

from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from mcp.types import CallToolResult, TextContent


@pytest.fixture
def mock_mcp_config(monkeypatch):
    monkeypatch.setattr("overwatchlooker.mcp_client.MCP_URL", "http://test:8080/mcp")
    monkeypatch.setattr("overwatchlooker.mcp_client.MCP_SOURCE", "test-source")


@pytest.fixture
def match_data():
    return {
        "map_name": "Numbani",
        "duration": "8:00",
        "mode": "ESCORT",
        "queue_type": "COMPETITIVE",
        "result": "VICTORY",
        "players": [],
    }


def _make_result(text: str = "OK", is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=is_error,
    )


def _patch_mcp_session(tool_result: CallToolResult):
    """Patch streamablehttp_client and ClientSession to return a given result."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=tool_result)

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client_ctx = AsyncMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

    return (
        patch("overwatchlooker.mcp_client.streamablehttp_client", return_value=mock_client_ctx),
        patch("overwatchlooker.mcp_client.ClientSession", return_value=mock_session_ctx),
        mock_session,
    )


class TestSubmitMatch:
    def test_calls_tool(self, mock_mcp_config, match_data):
        result = _make_result()
        p1, p2, mock_session = _patch_mcp_session(result)
        with p1, p2:
            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data)

        mock_session.call_tool.assert_called_once()
        name, args = mock_session.call_tool.call_args[0]
        assert name == "submit_match"
        assert args["map_name"] == "Numbani"
        assert args["source"] == "test-source"

    def test_includes_screenshot(self, mock_mcp_config, match_data):
        result = _make_result()
        p1, p2, mock_session = _patch_mcp_session(result)
        with p1, p2:
            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data, png_bytes=b"\x89PNG_fake")

        args = mock_session.call_tool.call_args[0][1]
        assert "screenshot_uploads" in args
        assert len(args["screenshot_uploads"]) == 1

    def test_no_screenshot(self, mock_mcp_config, match_data):
        result = _make_result()
        p1, p2, mock_session = _patch_mcp_session(result)
        with p1, p2:
            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data, png_bytes=None)

        args = mock_session.call_tool.call_args[0][1]
        assert "screenshot_uploads" not in args

    def test_backfill_flag(self, mock_mcp_config, match_data):
        result = _make_result()
        p1, p2, mock_session = _patch_mcp_session(result)
        with p1, p2:
            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data, is_backfill=True)

        args = mock_session.call_tool.call_args[0][1]
        assert args["is_backfill"] is True

    def test_error_raises(self, mock_mcp_config, match_data):
        result = _make_result("something failed", is_error=True)
        p1, p2, _ = _patch_mcp_session(result)
        with p1, p2:
            from overwatchlooker.mcp_client import submit_match
            with pytest.raises(RuntimeError, match="MCP error"):
                submit_match(match_data)

    def test_no_url_raises(self, monkeypatch, match_data):
        monkeypatch.setattr("overwatchlooker.mcp_client.MCP_URL", "")
        from overwatchlooker.mcp_client import submit_match
        with pytest.raises(RuntimeError, match="MCP_URL not set"):
            submit_match(match_data)

    def test_rank_range(self, mock_mcp_config, match_data):
        match_data["rank_min"] = "Gold 3"
        match_data["rank_max"] = "Diamond 1"
        match_data["is_wide_match"] = True
        result = _make_result()
        p1, p2, mock_session = _patch_mcp_session(result)
        with p1, p2:
            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data)

        args = mock_session.call_tool.call_args[0][1]
        assert args["rank_min"] == "Gold 3"
        assert args["rank_max"] == "Diamond 1"
        assert args["is_wide_match"] is True

    def test_banned_heroes(self, mock_mcp_config, match_data):
        match_data["banned_heroes"] = ["Mercy", "Zarya"]
        result = _make_result()
        p1, p2, mock_session = _patch_mcp_session(result)
        with p1, p2:
            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data)

        args = mock_session.call_tool.call_args[0][1]
        assert args["banned_heroes"] == ["Mercy", "Zarya"]


class TestExtractMatchId:
    def test_json_id_in_text(self):
        from overwatchlooker.mcp_client import _extract_match_id
        result = _make_result(json.dumps({"id": "abc-123-def"}))
        assert _extract_match_id(result) == "abc-123-def"

    def test_plain_text_no_id(self):
        from overwatchlooker.mcp_client import _extract_match_id
        result = _make_result("Match created: 812eec68-22c0-4786-aa73-4c50c18b14b7")
        assert _extract_match_id(result) is None

    def test_no_id(self):
        from overwatchlooker.mcp_client import _extract_match_id
        result = _make_result("Done")
        assert _extract_match_id(result) is None

    def test_structured_content(self):
        from overwatchlooker.mcp_client import _extract_match_id
        result = CallToolResult(
            content=[TextContent(type="text", text="OK")],
            structuredContent={"id": "structured-uuid-here"},
        )
        assert _extract_match_id(result) == "structured-uuid-here"
