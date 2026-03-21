"""Generate synthetic training data for hero panel value OCR (v3).

Futura font, digits + % + , + . only, white on black.
Heavy oversampling of hard cases (0, 1, comma numbers).

Usage:
    uv run python tools/generate_value_training.py --font /path/to/Futura.ttf [--count N] [--output DIR]
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = None  # Set via --font argument

# Characters: digits, percent, comma, period, colon (for timers like 00:00)
CHARS = "0123456789%,.:"

# Font sizes: 46 is 4K match, vary ±30%
FONT_SIZES = [32, 34, 36, 38, 40, 42, 44, 46, 46, 46, 48, 50, 52, 56, 60]


def random_value() -> str:
    """Generate a random stat value string with heavy hard-case oversampling."""
    r = random.random()
    if r < 0.15:
        # Zero — most common single value, was problematic
        return "0"
    elif r < 0.25:
        # Single digit 1-9 — "1" especially problematic (thin glyph)
        return str(random.randint(1, 9))
    elif r < 0.40:
        # Comma numbers — the main failure case (comma read as period)
        n = random.randint(1000, 99999)
        return f"{n:,}"
    elif r < 0.52:
        # Percentage 0%-100%
        return f"{random.randint(0, 100)}%"
    elif r < 0.60:
        # Small number 2-30
        return str(random.randint(2, 30))
    elif r < 0.68:
        # Three-digit number (100-999)
        return str(random.randint(100, 999))
    elif r < 0.73:
        # Large comma number (100,000+)
        n = random.randint(100000, 999999)
        return f"{n:,}"
    elif r < 0.78:
        # Two digit number
        return str(random.randint(10, 99))
    elif r < 0.83:
        # Decimal percentage like 47.5%
        return f"{random.uniform(0, 100):.1f}%"
    elif r < 0.88:
        # Just "1" — extra weight for the thin glyph
        return "1"
    elif r < 0.90:
        # Specific hard comma patterns
        return random.choice([
            "1,000", "2,470", "1,789", "2,035", "10,000", "1,111",
            "12,345", "99,999", "1,001", "2,000", "5,614", "3,750",
            "7,379", "4,753", "11,074", "14,886",
        ])
    elif r < 0.95:
        # Timer values (M:SS or MM:SS or 00:00)
        m = random.randint(0, 20)
        s = random.randint(0, 59)
        if random.random() < 0.3:
            return f"{m:02d}:{s:02d}"
        else:
            return f"{m}:{s:02d}"
    else:
        # Random digit string
        length = random.randint(1, 6)
        return "".join(random.choice("0123456789") for _ in range(length))


def render_sample(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render white text on black background."""
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x = random.randint(4, 20)
    pad_y = random.randint(4, 12)
    img_w = text_w + pad_x * 2
    img_h = text_h + pad_y * 2

    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=(255, 255, 255), font=font)

    return img


def generate_charset_coverage(fonts, output_dir):
    """Ensure every character appears at multiple sizes + hard confusable pairs."""
    lines = []
    idx = 0

    # Individual characters at multiple sizes
    for ch in CHARS:
        for font in random.sample(list(fonts.values()), min(5, len(fonts))):
            img = render_sample(ch, font)
            fname = f"value_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{ch}")
            idx += 1

    # Confusable / hard sequences at multiple sizes
    hard_cases = [
        "0", "00", "000", "0,000", "10,000", "100,000",
        "1", "11", "111", "1,111", "1,000", "10,001",
        "0%", "1%", "10%", "100%", "50%",
        "0.0%", "1.0%", "0.1%", "99.9%",
        ",", ".", "0,0", "0.0",
        "1,789", "2,470", "2,035", "12,345",
        "1.", "1,", ".1", ",1",
        "0:00", "00:00", "1:00", "1:23", "10:00", "12:34",
        "0:01", "0:59", "9:99", "1:01",
        "0:00", "0.00", "1:00", "1.00",  # colon vs period
    ]
    for text in hard_cases:
        for font in random.sample(list(fonts.values()), min(4, len(fonts))):
            img = render_sample(text, font)
            fname = f"value_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{text}")
            idx += 1

    return lines, idx


def main():
    parser = argparse.ArgumentParser(description="Generate panel value training data (v3)")
    parser.add_argument("--font", type=str, required=True,
                        help="Path to Futura.ttf font file")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("training_data/panel_values"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global FONT_PATH
    FONT_PATH = args.font

    random.seed(args.seed)
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
