"""Generate synthetic training data for ATTACK/DEFEND team side OCR.

Big Noodle Titling Oblique font, restricted charset (ATACKDEFN),
with varied backgrounds to handle different map skyboxes/lighting.
Text is semi-transparent with colored border, matching in-game rendering.

Usage:
    uv run python tools/generate_side_training.py --font /path/to/BigNoodleTooOblique.ttf [--count N] [--output DIR]
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

FONT_PATH = None  # Set via --font argument

# Only two words we need to recognize
WORDS = ["ATTACK", "DEFEND"]

# Text colors (RGB) — the fill color of the letters
ATTACK_COLOR = (199, 13, 78)
DEFEND_COLOR = (137, 233, 253)

# Border/outline colors (RGB)
ATTACK_BORDER = (167, 60, 68)
DEFEND_BORDER = (123, 185, 208)

# Font sizes matching the in-game text at 1080p (the ROI is 576x216)
FONT_SIZES = [34, 36, 38, 40, 42, 44, 46]

# Output image size — tight crop around text, not the full ROI
# The text is ~160px wide, ~50px tall at 1080p. Add padding for variation.
IMG_W = 280
IMG_H = 80


def random_background(w: int, h: int) -> Image.Image:
    """Generate a random background simulating varied map skyboxes."""
    r = random.random()

    if r < 0.25:
        # Solid color with slight noise
        base = (random.randint(20, 220), random.randint(20, 220), random.randint(20, 220))
        arr = np.full((h, w, 3), base, dtype=np.uint8)
        noise = np.random.randint(-15, 16, (h, w, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif r < 0.50:
        # Horizontal gradient
        c1 = [random.randint(20, 220) for _ in range(3)]
        c2 = [random.randint(20, 220) for _ in range(3)]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            t = x / w
            for c in range(3):
                arr[:, x, c] = int(c1[c] * (1 - t) + c2[c] * t)
        noise = np.random.randint(-10, 11, (h, w, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif r < 0.70:
        # Blue sky-like (the problematic case)
        v_base = random.randint(200, 250)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / h
            brightness = int(v_base * (1 - t * 0.3))
            arr[y, :] = (
                int(brightness * 0.7),
                int(brightness * 0.85),
                brightness,
            )
        noise = np.random.randint(-8, 9, (h, w, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif r < 0.85:
        # Dark/warm tones (indoor maps, sunset)
        base_r = random.randint(60, 180)
        base_g = random.randint(40, 140)
        base_b = random.randint(30, 120)
        arr = np.full((h, w, 3), (base_r, base_g, base_b), dtype=np.uint8)
        noise = np.random.randint(-20, 21, (h, w, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    else:
        # Random patchy (simulates buildings/structures)
        arr = np.random.randint(30, 200, (h // 8, w // 8, 3), dtype=np.uint8)
        img = Image.fromarray(arr).resize((w, h), Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(radius=4))
        return img


def render_sample(word: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render ATTACK or DEFEND on a random background with semi-transparency."""
    bg = random_background(IMG_W, IMG_H)

    # Text positioning — centered-ish with some variation
    bbox = font.getbbox(word)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = random.randint(10, max(10, IMG_W - text_w - 10))
    text_y = random.randint(10, max(10, IMG_H - text_h - 10))

    # Get text color and border
    if word == "ATTACK":
        fill_color = ATTACK_COLOR
        border_color = ATTACK_BORDER
    else:
        fill_color = DEFEND_COLOR
        border_color = DEFEND_BORDER

    # Vary the colors slightly for robustness
    fill_color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in fill_color)
    border_color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in border_color)

    # Semi-transparent text: draw on a separate layer and blend
    text_layer = Image.new("RGBA", (IMG_W, IMG_H), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)

    # Draw border (outline) by drawing text offset in multiple directions
    border_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in border_offsets:
        text_draw.text((text_x + dx, text_y + dy), word,
                       fill=(*border_color, 255), font=font)

    # Draw main text
    alpha = random.randint(180, 240)  # semi-transparent
    text_draw.text((text_x, text_y), word,
                   fill=(*fill_color, alpha), font=font)

    # Composite
    bg = bg.convert("RGBA")
    result = Image.alpha_composite(bg, text_layer)
    return result.convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ATTACK/DEFEND training data")
    parser.add_argument("--font", type=str, required=True,
                        help="Path to BigNoodleTooOblique.ttf")
    parser.add_argument("--count", type=int, default=3000,
                        help="Samples per word (total = count * 2)")
    parser.add_argument("--output", type=Path,
                        default=Path("training_data/team_side"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global FONT_PATH
    FONT_PATH = args.font

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    fonts = {sz: ImageFont.truetype(FONT_PATH, sz) for sz in FONT_SIZES}

    train_lines = []
    idx = 0

    for word in WORDS:
        print(f"Generating {args.count} samples for '{word}'...")
        for i in range(args.count):
            font = fonts[random.choice(FONT_SIZES)]
            img = render_sample(word, font)
            fname = f"side_{idx:06d}.png"
            img.save(args.output / fname)
            train_lines.append(f"{fname}\t{word}")
            idx += 1

            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{args.count}")

    # Write char dict
    chars = sorted(set("".join(WORDS)))
    dict_path = args.output / "dict.txt"
    dict_path.write_text("\n".join(chars), encoding="utf-8")

    # Split 90/10 train/val
    random.shuffle(train_lines)
    split = int(len(train_lines) * 0.9)
    (args.output / "train_list.txt").write_text(
        "\n".join(train_lines[:split]), encoding="utf-8")
    (args.output / "val_list.txt").write_text(
        "\n".join(train_lines[split:]), encoding="utf-8")

    print(f"\nDone: {len(train_lines)} total samples in {args.output}")
    print(f"  Train: {split}, Val: {len(train_lines) - split}")
    print(f"  Dict: {dict_path} ({len(chars)} chars: {''.join(chars)})")


if __name__ == "__main__":
    main()
