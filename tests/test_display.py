"""Tests for display formatting."""

from overwatchlooker.display import SEPARATOR, format_analysis, print_analysis


class TestFormatAnalysis:
    def test_has_timestamp(self):
        result = format_analysis("test content")
        assert "Analysis at" in result

    def test_wraps_content(self):
        result = format_analysis("MAP: Numbani")
        assert "MAP: Numbani" in result

    def test_has_separators(self):
        result = format_analysis("test")
        assert result.startswith(SEPARATOR)
        assert result.endswith(SEPARATOR)


class TestPrintAnalysis:
    def test_returns_formatted(self):
        result = print_analysis("test content")
        assert "test content" in result
        assert "Analysis at" in result
