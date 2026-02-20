import datetime

SEPARATOR = "=" * 60


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
    print(f"\n{formatted}\n")
    return formatted


def print_status(message: str) -> None:
    """Print a status message."""
    print(f"[OWL] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"[OWL ERROR] {message}")
