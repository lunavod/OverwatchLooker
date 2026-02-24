import datetime
import logging
import sys
from pathlib import Path

SEPARATOR = "=" * 60

# File logger that works even with pythonw (no stdout)
_logs_dir = Path(__file__).parent.parent / "logs"
_logs_dir.mkdir(exist_ok=True)
_log_path = _logs_dir / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    filename=str(_log_path),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger("overwatchlooker")


def _safe_print(msg: str) -> None:
    """Print that silently does nothing when stdout is unavailable (pythonw)."""
    _logger.info(msg)
    try:
        print(msg)
    except Exception:
        pass


def format_analysis(analysis: str) -> str:
    """Format the analysis result with a timestamp header and footer."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{SEPARATOR}\n"
        f"  OVERWATCH LOOKER -- Analysis at {now}\n"
        f"{SEPARATOR}\n"
        f"{analysis}\n"
        f"{SEPARATOR}"
    )


def print_analysis(analysis: str) -> str:
    """Print the analysis result and return the formatted text."""
    formatted = format_analysis(analysis)
    _safe_print(f"\n{formatted}\n")
    return formatted


def print_status(message: str) -> None:
    """Print a status message."""
    _safe_print(f"[OWL] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    _safe_print(f"[OWL ERROR] {message}")
