import os

from dotenv import load_dotenv

load_dotenv()

# Display / capture
MONITOR_INDEX: int = 1  # mss monitor index (1 = primary display)

# Subtitle listener settings
SUBTITLE_POLL_INTERVAL: float = 1.0  # seconds between subtitle region checks
AUDIO_COOLDOWN_SECONDS: float = 30.0  # minimum seconds between detections

# Player identity
OVERWATCH_USERNAME: str = os.environ.get("OVERWATCH_USERNAME", "")

# MCP server
MCP_URL: str = os.environ.get("MCP_URL", "")
MCP_SOURCE: str = os.environ.get("MCP_SOURCE", "looker")

# WebSocket server for companion app
WS_PORT: int = int(os.environ.get("WS_PORT", "42685"))

# Overwolf receiver (OverwatchListener connects here)
OVERWOLF_PORT: int = int(os.environ.get("OVERWOLF_PORT", "28025"))

# Auto-recording cleanup
RECORDINGS_KEEP: int = int(os.environ.get("RECORDINGS_KEEP", "5"))

# LLM fallback analyzer (used when Overwolf GEP is unavailable)
CODEX_MODEL: str = os.environ.get("CODEX_MODEL", "gpt-5.3-codex")
CODEX_REASONING: str | None = os.environ.get("CODEX_REASONING", None) or None
