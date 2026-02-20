import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
AWS_KEY: str = os.environ.get("AWS_KEY", "")
AWS_SECRET: str = os.environ.get("AWS_SECRET", "")
AWS_REGION: str = os.environ.get("AWS_REGION", "eu-west-1")
ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
BEDROCK_MODEL: str = "eu.anthropic.claude-sonnet-4-6"
MONITOR_INDEX: int = 1  # mss monitor index (1 = primary display)
MAX_TOKENS: int = 16000
