"""Tests for MCP client: HTTP protocol, JSON-RPC format."""

from unittest.mock import MagicMock, patch

import pytest


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


class TestSubmitMatch:
    def test_sends_jsonrpc(self, mock_mcp_config, match_data):
        with patch("overwatchlooker.mcp_client.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            # Mock responses for initialize, notification, and tool call
            init_resp = MagicMock()
            init_resp.headers = {"mcp-session-id": "test-session"}
            notif_resp = MagicMock()
            tool_resp = MagicMock()
            tool_resp.json.return_value = {"result": {"id": 1}}

            mock_client.post.side_effect = [init_resp, notif_resp, tool_resp]

            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data)

            # Verify 3 POST calls: initialize, notification, tool call
            assert mock_client.post.call_count == 3

            # Verify tool call payload is JSON-RPC 2.0
            tool_call = mock_client.post.call_args_list[2]
            payload = tool_call.kwargs.get("json") or tool_call[1].get("json")
            assert payload["jsonrpc"] == "2.0"
            assert payload["method"] == "tools/call"
            assert payload["params"]["name"] == "submit_match"

    def test_includes_screenshot(self, mock_mcp_config, match_data):
        with patch("overwatchlooker.mcp_client.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            init_resp = MagicMock()
            init_resp.headers = {"mcp-session-id": "s"}
            tool_resp = MagicMock()
            tool_resp.json.return_value = {"result": {}}
            mock_client.post.side_effect = [init_resp, MagicMock(), tool_resp]

            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data, png_bytes=b"\x89PNG_fake")

            tool_call = mock_client.post.call_args_list[2]
            payload = tool_call.kwargs.get("json") or tool_call[1].get("json")
            args = payload["params"]["arguments"]
            assert "screenshot_uploads" in args
            assert len(args["screenshot_uploads"]) == 1

    def test_no_screenshot(self, mock_mcp_config, match_data):
        with patch("overwatchlooker.mcp_client.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            init_resp = MagicMock()
            init_resp.headers = {}
            tool_resp = MagicMock()
            tool_resp.json.return_value = {"result": {}}
            mock_client.post.side_effect = [init_resp, MagicMock(), tool_resp]

            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data, png_bytes=None)

            tool_call = mock_client.post.call_args_list[2]
            payload = tool_call.kwargs.get("json") or tool_call[1].get("json")
            args = payload["params"]["arguments"]
            assert "screenshot_uploads" not in args

    def test_session_header(self, mock_mcp_config, match_data):
        with patch("overwatchlooker.mcp_client.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            init_resp = MagicMock()
            init_resp.headers = {"mcp-session-id": "my-session-123"}
            tool_resp = MagicMock()
            tool_resp.json.return_value = {"result": {}}
            mock_client.post.side_effect = [init_resp, MagicMock(), tool_resp]

            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data)

            # Tool call should include session header
            tool_call = mock_client.post.call_args_list[2]
            headers = tool_call.kwargs.get("headers") or tool_call[1].get("headers")
            assert headers["mcp-session-id"] == "my-session-123"

    def test_backfill_flag(self, mock_mcp_config, match_data):
        with patch("overwatchlooker.mcp_client.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            init_resp = MagicMock()
            init_resp.headers = {}
            tool_resp = MagicMock()
            tool_resp.json.return_value = {"result": {}}
            mock_client.post.side_effect = [init_resp, MagicMock(), tool_resp]

            from overwatchlooker.mcp_client import submit_match
            submit_match(match_data, is_backfill=True)

            tool_call = mock_client.post.call_args_list[2]
            payload = tool_call.kwargs.get("json") or tool_call[1].get("json")
            assert payload["params"]["arguments"]["is_backfill"] is True

    def test_error_raises(self, mock_mcp_config, match_data):
        with patch("overwatchlooker.mcp_client.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            init_resp = MagicMock()
            init_resp.headers = {}
            tool_resp = MagicMock()
            tool_resp.json.return_value = {"error": {"code": -1, "message": "fail"}}
            mock_client.post.side_effect = [init_resp, MagicMock(), tool_resp]

            from overwatchlooker.mcp_client import submit_match
            with pytest.raises(RuntimeError, match="MCP error"):
                submit_match(match_data)

    def test_no_url_raises(self, monkeypatch, match_data):
        monkeypatch.setattr("overwatchlooker.mcp_client.MCP_URL", "")
        from overwatchlooker.mcp_client import submit_match
        with pytest.raises(RuntimeError, match="MCP_URL not set"):
            submit_match(match_data)
