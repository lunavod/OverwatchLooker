"""Analyzer registry — resolves ANALYZER config to the right backend."""

from overwatchlooker.config import ANALYZER

VALID_ANALYZERS = ("anthropic", "codex", "ocr")


def get_analyze_screenshot():
    """Return the analyze_screenshot function for the configured analyzer."""
    if ANALYZER == "anthropic":
        from overwatchlooker.analyzers.anthropic import analyze_screenshot
    elif ANALYZER == "codex":
        from overwatchlooker.analyzers.codex import analyze_screenshot
    elif ANALYZER == "ocr":
        from overwatchlooker.analyzers.ocr import analyze_screenshot
    else:
        raise ValueError(
            f"Unknown ANALYZER={ANALYZER!r}. Valid options: {', '.join(VALID_ANALYZERS)}"
        )
    return analyze_screenshot
