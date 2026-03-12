"""Tests for screenshot processing: validation, hero panel detection, OCR."""

import cv2
import numpy as np
import pytest

from overwatchlooker.heroes import ALL_HEROES
from overwatchlooker.screenshot import (
    HERO_PANEL_REGION,
    crop_hero_panel,
    has_hero_panel,
    is_ow2_tab_screen,
    ocr_hero_name,
    resize_for_analyzer,
)

from tests.conftest import FIXTURES_DIR, SCREENSHOT_FIXTURES


# --- is_ow2_tab_screen ---

@pytest.mark.screenshots
class TestIsOw2TabScreen:
    def test_valid_tab_screen(self, sample_tab_png):
        result, reason = is_ow2_tab_screen(sample_tab_png)
        assert result
        assert reason == ""

    def test_invalid_non_tab(self, sample_non_tab_png):
        result, reason = is_ow2_tab_screen(sample_non_tab_png)
        assert not result
        assert len(reason) > 0

    def test_returns_tuple(self, sample_tab_png):
        result = is_ow2_tab_screen(sample_tab_png)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_invalid_bytes(self):
        result, reason = is_ow2_tab_screen(b"not an image")
        assert not result

    def test_all_tab_screenshots_valid(self, fixtures_dir):
        """All fixture screenshots that have hero panels should be valid tab screens."""
        for filename, has_panel, _ in SCREENSHOT_FIXTURES:
            if has_panel:
                png = (fixtures_dir / filename).read_bytes()
                result, reason = is_ow2_tab_screen(png)
                assert result, f"{filename} should be valid Tab: {reason}"


# --- has_hero_panel ---

@pytest.mark.screenshots
class TestHasHeroPanel:
    def test_positive(self, sample_hero_panel_png):
        assert has_hero_panel(sample_hero_panel_png)

    def test_negative(self, sample_no_panel_png):
        assert not has_hero_panel(sample_no_panel_png)

    def test_invalid_bytes(self):
        assert not has_hero_panel(b"garbage")

    def test_all_fixtures(self, fixtures_dir):
        for filename, expected_panel, _ in SCREENSHOT_FIXTURES:
            png = (fixtures_dir / filename).read_bytes()
            result = has_hero_panel(png)
            assert bool(result) == expected_panel, (
                f"{filename}: has_hero_panel={result}, expected={expected_panel}"
            )


# --- crop_hero_panel ---

@pytest.mark.screenshots
class TestCropHeroPanel:
    def test_returns_valid_png(self, sample_hero_panel_png):
        crop = crop_hero_panel(sample_hero_panel_png)
        assert crop[:4] == b"\x89PNG"

    def test_crop_smaller_than_original(self, sample_hero_panel_png):
        crop = crop_hero_panel(sample_hero_panel_png)
        assert len(crop) < len(sample_hero_panel_png)

    def test_crop_dimensions(self, sample_hero_panel_png):
        """Crop should be roughly 31% width x 73% height of original."""
        orig = cv2.imdecode(
            np.frombuffer(sample_hero_panel_png, np.uint8), cv2.IMREAD_COLOR
        )
        crop_bytes = crop_hero_panel(sample_hero_panel_png)
        crop = cv2.imdecode(
            np.frombuffer(crop_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        oh, ow = orig.shape[:2]
        ch, cw = crop.shape[:2]
        # HERO_PANEL_REGION = (0.60, 0.12, 0.91, 0.85) → width~31%, height~73%
        assert 0.25 < cw / ow < 0.40
        assert 0.65 < ch / oh < 0.80


# --- ocr_hero_name ---

@pytest.mark.screenshots
class TestOcrHeroName:
    @pytest.mark.parametrize("filename,expected_hero", [
        ("2026-03-07_21-33-27.png", "Juno"),
        ("2026-03-07_21-00-39.png", "Reinhardt"),
        ("2026-03-11_23-37-05.png", "Moira"),
    ])
    def test_reads_hero_name(self, fixtures_dir, filename, expected_hero):
        png = (fixtures_dir / filename).read_bytes()
        crop = crop_hero_panel(png)
        name = ocr_hero_name(crop)
        assert name == expected_hero, f"Expected '{expected_hero}', got '{name}'"

    def test_result_is_canonical(self, fixtures_dir):
        """OCR result should always be in ALL_HEROES or empty."""
        for filename, has_panel, _ in SCREENSHOT_FIXTURES:
            if not has_panel:
                continue
            png = (fixtures_dir / filename).read_bytes()
            crop = crop_hero_panel(png)
            name = ocr_hero_name(crop)
            assert name == "" or name in ALL_HEROES, (
                f"{filename}: '{name}' not in ALL_HEROES"
            )


# --- resize_for_analyzer ---

class TestResizeForAnalyzer:
    def _make_png(self, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def test_shrinks_large_image(self):
        big = self._make_png(3840, 2160)
        small = resize_for_analyzer(big, "anthropic")
        assert len(small) < len(big)

    def test_no_grow_small_image(self):
        small_png = self._make_png(800, 600)
        result = resize_for_analyzer(small_png, "anthropic")
        assert result is small_png  # identity, not just equal

    def test_no_resize_for_codex(self):
        big = self._make_png(3840, 2160)
        result = resize_for_analyzer(big, "codex")
        assert result is big

