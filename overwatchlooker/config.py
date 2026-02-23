import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
VISION_MODEL: str = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")
MONITOR_INDEX: int = 1  # mss monitor index (1 = primary display)
MAX_TOKENS: int = 16000
