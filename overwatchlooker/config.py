import os

from dotenv import load_dotenv

load_dotenv()

# Analyzer backend: "claude" or "ocr"
ANALYZER: str = os.environ.get("ANALYZER", "claude")

# Claude Vision settings
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS: int = 16000

# Display
MONITOR_INDEX: int = 1  # mss monitor index (1 = primary display)

# Audio listener settings
AUDIO_CHUNK_DURATION: float = 4.0       # seconds of audio in ring buffer
AUDIO_HOP_DURATION: float = 0.5         # seconds between processing steps
AUDIO_COOLDOWN_SECONDS: float = 30.0    # minimum seconds between detections
AUDIO_MATCH_THRESHOLD: float = 0.25     # 1D NCC threshold (true match ~0.4, noise <0.08)
AUDIO_MATCH_MARGIN: float = 0.10        # winner must beat runner-up by this much
AUDIO_MIN_RMS: float = 0.0005           # minimum RMS energy to attempt matching
SCREENSHOT_MAX_AGE_SECONDS: float = 120.0  # max age of screenshot to analyze

# Telegram
TELEGRAM_TOKEN: str = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHANNEL: str = os.environ.get("TELEGRAM_CHANNEL", "")
