"""End-to-end test for per-hero screenshot tracking pipeline.

Uses real screenshots from Pictures/OverwatchLooker to test:
1. has_hero_panel() detection
2. crop_hero_panel() cropping
3. ocr_hero_name() OCR
4. Hero crop deduplication
5. Final hero removal before analysis
"""

from pathlib import Path

from overwatchlooker.screenshot import (
    crop_hero_panel,
    has_hero_panel,
    ocr_hero_name,
)
from overwatchlooker.heroes import edit_distance as _edit_distance

SCREENSHOTS_DIR = Path(r"C:\Users\yegor\OneDrive\Pictures\OverwatchLooker")

# (filename, expected_has_panel, expected_hero_name_substring_or_None)
TEST_CASES = [
    ("2026-03-07_21-33-27.png", True, "Juno"),       # Tab 1: Juno, narrow scoreboard
    ("2026-03-07_21-37-05.png", True, "Juno"),       # Tab 2: Juno, medium (should dedup)
    ("2026-03-07_21-41-45.png", True, "Juno"),       # Tab 3: Juno, wide (should dedup)
    ("2026-03-07_21-00-39.png", True, "Reinhardt"),  # Tab 4: Reinhardt
    ("2026-03-11_22-15-07.png", False, None),        # Tab 6: match end, no panel
    ("2026-03-11_23-37-05.png", True, "Moira"),      # Tab 7: Moira (final screenshot)
]


def test_hero_panel_detection():
    """Test that has_hero_panel correctly identifies hero panels."""
    print("=== Test: has_hero_panel ===")
    for filename, expected, _ in TEST_CASES:
        path = SCREENSHOTS_DIR / filename
        if not path.exists():
            print(f"  SKIP (missing): {filename}")
            continue
        png_bytes = path.read_bytes()
        result = has_hero_panel(png_bytes)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: {filename} -> has_panel={result} (expected={expected})")
        assert result == expected, f"{filename}: expected {expected}, got {result}"


def test_hero_name_ocr():
    """Test that ocr_hero_name reads correct hero names."""
    print("\n=== Test: ocr_hero_name ===")
    for filename, has_panel, expected_name in TEST_CASES:
        if not has_panel or expected_name is None:
            continue
        path = SCREENSHOTS_DIR / filename
        if not path.exists():
            print(f"  SKIP (missing): {filename}")
            continue
        png_bytes = path.read_bytes()
        crop = crop_hero_panel(png_bytes)
        name = ocr_hero_name(crop)
        # Fuzzy match: edit distance <= 2
        match = _edit_distance(name.lower(), expected_name.lower()) <= 2 if name else False
        status = "PASS" if match else "FAIL"
        print(f"  {status}: {filename} -> '{name}' (expected ~'{expected_name}')")
        assert match, f"{filename}: expected ~'{expected_name}', got '{name}'"


def test_dedup_pipeline():
    """Test the full dedup pipeline: simulated Tab presses."""
    print("\n=== Test: dedup pipeline ===")
    hero_crops: dict[str, bytes] = {}

    # Simulate Tab presses in order
    tab_sequence = [
        ("2026-03-07_21-33-27.png", "Juno"),       # store
        ("2026-03-07_21-37-05.png", "Juno"),       # dedup skip
        ("2026-03-07_21-41-45.png", "Juno"),       # dedup skip
        ("2026-03-07_21-00-39.png", "Reinhardt"),  # store
        ("2026-03-07_21-00-39.png", "Reinhardt"),  # dedup skip (duplicate)
        ("2026-03-11_22-15-07.png", None),         # no panel
    ]

    for filename, expected_hero in tab_sequence:
        path = SCREENSHOTS_DIR / filename
        if not path.exists():
            print(f"  SKIP (missing): {filename}")
            continue
        png_bytes = path.read_bytes()
        if has_hero_panel(png_bytes):
            crop = crop_hero_panel(png_bytes)
            name = ocr_hero_name(crop)
            if name:
                if not any(_edit_distance(name.lower(), k.lower()) <= 2 for k in hero_crops):
                    hero_crops[name] = crop
                    print(f"  STORED: {filename} -> '{name}'")
                else:
                    print(f"  DEDUP SKIP: {filename} -> '{name}'")
            else:
                print(f"  OCR FAIL: {filename}")
        else:
            print(f"  NO PANEL: {filename}")

    print(f"\n  Hero crops: {list(hero_crops.keys())}")
    assert len(hero_crops) == 2, f"Expected 2 hero crops, got {len(hero_crops)}: {list(hero_crops.keys())}"

    # Verify Juno and Reinhardt are present (fuzzy)
    names_lower = [k.lower() for k in hero_crops]
    assert any(_edit_distance("juno", n) <= 2 for n in names_lower), "Missing Juno in crops"
    assert any(_edit_distance("reinhardt", n) <= 2 for n in names_lower), "Missing Reinhardt in crops"
    print("  PASS: 2 hero crops (Juno + Reinhardt)")


def test_final_hero_removal():
    """Test that the final screenshot's hero is removed from crops."""
    print("\n=== Test: final hero removal ===")
    # Build crops as in dedup test
    hero_crops: dict[str, bytes] = {}
    for filename in ["2026-03-07_21-33-27.png", "2026-03-07_21-00-39.png", "2026-03-11_23-37-05.png"]:
        path = SCREENSHOTS_DIR / filename
        if not path.exists():
            print(f"  SKIP (missing): {filename}")
            return
        png_bytes = path.read_bytes()
        if has_hero_panel(png_bytes):
            crop = crop_hero_panel(png_bytes)
            name = ocr_hero_name(crop)
            if name and not any(_edit_distance(name.lower(), k.lower()) <= 2 for k in hero_crops):
                hero_crops[name] = crop

    print(f"  Before removal: {list(hero_crops.keys())}")

    # Simulate final screenshot = Moira
    final_path = SCREENSHOTS_DIR / "2026-03-11_23-37-05.png"
    final_bytes = final_path.read_bytes()
    final_crop = crop_hero_panel(final_bytes)
    final_hero = ocr_hero_name(final_crop)
    print(f"  Final hero: {final_hero}")

    # Remove final hero from crops
    to_remove = [k for k in hero_crops if _edit_distance(final_hero.lower(), k.lower()) <= 2]
    for k in to_remove:
        del hero_crops[k]

    print(f"  After removal: {list(hero_crops.keys())}")
    # Moira should be removed, leaving Juno + Reinhardt
    assert not any(_edit_distance("moira", k.lower()) <= 2 for k in hero_crops), "Moira should be removed"
    assert len(hero_crops) == 2, f"Expected 2 crops after removal, got {len(hero_crops)}"
    print("  PASS: Moira removed, 2 crops remain")


if __name__ == "__main__":
    test_hero_panel_detection()
    test_hero_name_ocr()
    test_dedup_pipeline()
    test_final_hero_removal()
    print("\n=== ALL TESTS PASSED ===")
