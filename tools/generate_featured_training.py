"""Generate synthetic training data for hero panel featured value OCR.

Noodle font (Big Noodle Titling or similar), digits + % + , + . + : only,
white on black. Heavy oversampling of timer values (MM:SS) since the
featured stat often shows objective contest time.

Usage:
    uv run python tools/generate_featured_training.py --font /path/to/Noodle.ttf [--count N] [--output DIR]
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = None  # Set via --font argument

# Characters: digits, percent, comma, period, colon
CHARS = "0123456789%,.:"

# Font sizes: featured value is large (~80px at 4K), vary for robustness
FONT_SIZES = [48, 52, 56, 60, 64, 68, 72, 76, 80, 80, 80, 84, 88, 92, 96]


def random_value() -> str:
    """Generate a random featured stat value string.

    Featured stats are typically: timer values (01:09), single numbers (5),
    percentages (29%), or comma numbers (1,234).
    Timer values are heavily oversampled since those are the hardest case.
    """
    r = random.random()
    if r < 0.30:
        # Timer values MM:SS — the main failure case
        m = random.randint(0, 20)
        s = random.randint(0, 59)
        if random.random() < 0.5:
            return f"{m:02d}:{s:02d}"
        else:
            return f"{m}:{s:02d}"
    elif r < 0.42:
        # Single digit 0-9
        return str(random.randint(0, 9))
    elif r < 0.52:
        # Small number 10-99
        return str(random.randint(10, 99))
    elif r < 0.62:
        # Percentage 0%-100%
        return f"{random.randint(0, 100)}%"
    elif r < 0.72:
        # Comma numbers
        n = random.randint(1000, 99999)
        return f"{n:,}"
    elif r < 0.78:
        # Three-digit number
        return str(random.randint(100, 999))
    elif r < 0.83:
        # Decimal percentage
        return f"{random.uniform(0, 100):.1f}%"
    elif r < 0.88:
        # Just "0" or "1" — thin glyphs
        return random.choice(["0", "1"])
    elif r < 0.93:
        # Specific timer patterns
        return random.choice([
            "00:00", "01:09", "00:30", "02:45", "10:00", "05:23",
            "0:00", "1:23", "3:45", "12:34", "0:01", "0:59",
            "00:01", "00:59", "01:00", "01:30", "02:00", "03:00",
        ])
    else:
        # Random digit string
        length = random.randint(1, 5)
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
            fname = f"featured_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{ch}")
            idx += 1

    # Confusable / hard sequences
    hard_cases = [
        "0", "00", "000",
        "1", "11", "111",
        "0%", "1%", "10%", "100%", "50%",
        ",", ".", ":", "0,0", "0.0",
        "0:00", "00:00", "1:00", "1:23", "10:00", "12:34",
        "0:01", "0:09", "0:59", "1:01", "1:09", "01:09",
        "00:01", "00:09", "00:30", "01:00", "02:45", "05:23",
        "0:00", "0.00", "1:00", "1.00",  # colon vs period
        "01:09", "07:09", "01:00", "07:00",  # 0 vs 7 confusion
        "1,000", "10,000", "1,234", "12,345",
    ]
    for text in hard_cases:
        for font in random.sample(list(fonts.values()), min(4, len(fonts))):
            img = render_sample(text, font)
            fname = f"featured_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{text}")
            idx += 1

    return lines, idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate featured value training data (Noodle font)")
    parser.add_argument("--font", type=str, required=True,
                        help="Path to Noodle font .ttf file")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=Path,
                        default=Path("training_data/panel_featured"))
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
        fname = f"featured_{start_idx + i:06d}.png"
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
    (args.output / "train_list.txt").write_text(
        "\n".join(train_lines[:split]), encoding="utf-8")
    (args.output / "val_list.txt").write_text(
        "\n".join(train_lines[split:]), encoding="utf-8")

    print(f"\nDone: {len(train_lines)} total samples in {args.output}")
    print(f"  Train: {split}, Val: {len(train_lines) - split}")
    print(f"  Dict: {dict_path} ({len(CHARS)} chars)")


if __name__ == "__main__":
    main()
