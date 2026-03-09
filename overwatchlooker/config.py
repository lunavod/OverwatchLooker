import os

from dotenv import load_dotenv

load_dotenv()

# Analyzer backend: "anthropic", "codex", or "ocr"
_analyzer_raw = os.environ.get("ANALYZER", "anthropic")
# Backward compat: "claude" -> "anthropic"
ANALYZER: str = "anthropic" if _analyzer_raw == "claude" else _analyzer_raw

# Claude Vision settings
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS: int = 16000

# Codex (ChatGPT) settings
CODEX_MODEL: str = os.environ.get("CODEX_MODEL", "gpt-5.3-codex")
CODEX_REASONING: str | None = os.environ.get("CODEX_REASONING", None)  # "low", "medium", "high", "xhigh", or None

# Display
MONITOR_INDEX: int = 1  # mss monitor index (1 = primary display)

# Audio listener settings
AUDIO_CHUNK_DURATION: float = 4.0       # seconds of audio in ring buffer
AUDIO_HOP_DURATION: float = 0.5         # seconds between processing steps
AUDIO_COOLDOWN_SECONDS: float = 30.0    # minimum seconds between detections
AUDIO_MATCH_THRESHOLD: float = 0.25     # 1D NCC threshold (consecutive hops prevent false positives)
AUDIO_CONFIRM_HOPS: int = 2            # must exceed threshold for this many consecutive hops
AUDIO_MATCH_MARGIN: float = 0.10        # winner must beat runner-up by this much
AUDIO_MIN_RMS: float = 0.0005           # minimum RMS energy to attempt matching
SCREENSHOT_MAX_AGE_SECONDS: float = 120.0  # max age of screenshot to analyze

# Subtitle listener settings
SUBTITLE_POLL_INTERVAL: float = 1.0  # seconds between subtitle region checks

# Telegram (Telethon user client)
TELEGRAM_API_ID: str = os.environ.get("TELEGRAM_API_ID", "")
TELEGRAM_API_HASH: str = os.environ.get("TELEGRAM_API_HASH", "")
TELEGRAM_CHANNEL: str = os.environ.get("TELEGRAM_CHANNEL", "")

# Player identity
OVERWATCH_USERNAME: str = os.environ.get("OVERWATCH_USERNAME", "")

# MCP server
MCP_URL: str = os.environ.get("MCP_URL", "")
MCP_SOURCE: str = os.environ.get("MCP_SOURCE", "looker")
