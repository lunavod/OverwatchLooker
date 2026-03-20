"""Generate synthetic training data for hero panel value OCR.

Futura font, digits + % + , + . only.
Varied backgrounds and text colors for robustness.

Usage:
    uv run python tools/generate_value_training.py [--count N] [--output DIR]
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = r"C:\Users\yegor\Downloads\Futura.ttf"

# Characters: digits, percent, comma, period
CHARS = "0123456789%,."

# Font sizes: 46 is 4K match, vary ±30%
FONT_SIZES = [32, 34, 36, 38, 40, 42, 44, 46, 46, 46, 48, 50, 52, 56, 60]

# Background colors (RGB)
BG_COLORS = [
    (0, 0, 0),         # pure black (binarized)
    (0, 0, 0),         # weighted
    (0, 0, 0),         # weighted
    (14, 18, 32),      # actual panel dark
    (20, 25, 40),      # panel lighter
    (10, 12, 22),      # panel darker
    (25, 30, 45),      # panel variation
    (15, 15, 15),      # neutral dark gray
    (5, 8, 18),        # very dark blue
]

# Text colors — values are bright white in-game
TEXT_COLORS = [
    (255, 255, 255),   # pure white
    (255, 255, 255),   # weighted
    (255, 255, 255),   # weighted
    (245, 245, 245),   # actual value white
    (235, 235, 235),   # slightly dimmer
    (250, 250, 250),   # near-white
    (220, 220, 220),   # light gray
    (200, 210, 220),   # slight blue tint
]


def random_value() -> str:
    """Generate a random stat value string."""
    r = random.random()
    if r < 0.20:
        # Zero (very common in OW2)
        return "0"
    elif r < 0.30:
        # Small number 1-30
        return str(random.randint(1, 30))
    elif r < 0.45:
        # Percentage 0%-100%
        return f"{random.randint(0, 100)}%"
    elif r < 0.60:
        # Comma number (1,000 - 99,999)
        n = random.randint(1000, 99999)
        return f"{n:,}"
    elif r < 0.70:
        # Three-digit number (100-999)
        return str(random.randint(100, 999))
    elif r < 0.78:
        # Decimal percentage like 47.5%
        return f"{random.uniform(0, 100):.1f}%"
    elif r < 0.85:
        # Large comma number (100,000+)
        n = random.randint(100000, 999999)
        return f"{n:,}"
    elif r < 0.90:
        # Single digit
        return str(random.randint(0, 9))
    elif r < 0.95:
        # Two digit number
        return str(random.randint(10, 99))
    else:
        # Random digit string
        length = random.randint(1, 6)
        return "".join(random.choice("0123456789") for _ in range(length))


def _jitter_color(color: tuple, amount: int = 10) -> tuple:
    return tuple(max(0, min(255, c + random.randint(-amount, amount))) for c in color)


def _add_noise(img: Image.Image, intensity: float = 0.05) -> Image.Image:
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-int(intensity * 255), int(intensity * 255) + 1,
                              arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_sample(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render text with varied colors and optional noise."""
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x = random.randint(4, 20)
    pad_y = random.randint(4, 12)
    img_w = text_w + pad_x * 2
    img_h = text_h + pad_y * 2

    bg = _jitter_color(random.choice(BG_COLORS))
    text_color = _jitter_color(random.choice(TEXT_COLORS))

    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=text_color, font=font)

    # Occasional noise (30% chance)
    if random.random() < 0.3:
        img = _add_noise(img, intensity=random.uniform(0.02, 0.08))

    # Occasional slight blur (15% chance)
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))

    return img


def generate_charset_coverage(fonts, output_dir):
    """Ensure every character appears in training data at multiple sizes."""
    lines = []
    idx = 0
    # Individual characters
    for ch in CHARS:
        for font in random.sample(list(fonts.values()), min(4, len(fonts))):
            img = render_sample(ch, font)
            fname = f"value_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{ch}")
            idx += 1

    # Confusable sequences
    confusables = [
        "0", "00", "000", "0,000", "10,000", "100,000",
        "1", "11", "111", "1,111",
        "0%", "1%", "10%", "100%",
        "0.0%", "1.0%", "0.1%",
        ",", ".", "0,0", "0.0",
    ]
    for text in confusables:
        for font in random.sample(list(fonts.values()), min(3, len(fonts))):
            img = render_sample(text, font)
            fname = f"value_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{text}")
            idx += 1

    return lines, idx


def main():
    parser = argparse.ArgumentParser(description="Generate panel value training data")
    parser.add_argument("--count", type=int, default=69000)
    parser.add_argument("--output", type=Path, default=Path("training_data/panel_values_v2"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    fonts = {sz: ImageFont.truetype(FONT_PATH, sz) for sz in FONT_SIZES}

    # Charset coverage first
    print("Generating charset coverage samples...")
    train_lines, start_idx = generate_charset_coverage(fonts, args.output)
    print(f"  {len(train_lines)} charset coverage samples")

    # Main generation
    print(f"Generating {args.count} random samples...")
    for i in range(args.count):
        text = random_value()
        font = fonts[random.choice(FONT_SIZES)]
        img = render_sample(text, font)
        fname = f"value_{start_idx + i:06d}.png"
        img.save(args.output / fname)
        train_lines.append(f"{fname}\t{text}")

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{args.count}")

    # Write char dict
    dict_path = args.output / "dict.txt"
    dict_path.write_text("\n".join(list(CHARS)), encoding="utf-8")

    # Split 90/10 train/val
    random.shuffle(train_lines)
    split = int(len(train_lines) * 0.9)
    (args.output / "train_list.txt").write_text("\n".join(train_lines[:split]), encoding="utf-8")
    (args.output / "val_list.txt").write_text("\n".join(train_lines[split:]), encoding="utf-8")

    print(f"\nDone: {len(train_lines)} total samples in {args.output}")
    print(f"  Train: {split}, Val: {len(train_lines) - split}")
    print(f"  Dict: {dict_path} ({len(CHARS)} chars)")


if __name__ == "__main__":
    main()
