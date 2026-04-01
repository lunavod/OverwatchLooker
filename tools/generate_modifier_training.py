"""Generate synthetic training data for modifier text OCR.

Futura font, A-Z + space, uppercase only, white on black.
Content is a mix of real OW2 modifier names and random uppercase text.

Usage:
    uv run python tools/generate_modifier_training.py --font /path/to/Futura.ttf [--count N] [--output DIR]
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

MODIFIERS = [
    "VICTORY", "DEFEAT", "DRAW",
    "UPHILL BATTLE", "EXPECTED", "CONSOLATION", "REVERSAL",
    "WINNING STREAK", "LOSING STREAK",
    "WINNING TREND", "LOSING TREND",
    "CALIBRATION", "VOLATILE",
    "DEMOTION PROTECTION", "DEMOTION",
    "WIDE", "PRESSURE",
]

# Font sizes matching modifier text at 1080p
FONT_SIZES = [18, 20, 22, 24, 26, 26, 26, 28, 30, 32]


def random_text() -> str:
    """Generate varied uppercase text for training."""
    r = random.random()
    if r < 0.45:
        # Real modifier
        return random.choice(MODIFIERS)
    elif r < 0.60:
        # Single word from a modifier
        words = []
        for m in MODIFIERS:
            words.extend(m.split())
        return random.choice(words)
    elif r < 0.75:
        # Random single word (3-12 chars)
        length = random.randint(3, 12)
        return "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(length))
    elif r < 0.88:
        # Two random words
        w1 = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(3, 10)))
        w2 = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(3, 10)))
        return f"{w1} {w2}"
    else:
        # Three random words
        words = []
        for _ in range(3):
            w = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(2, 8)))
            words.append(w)
        return " ".join(words)


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
    """Ensure every character appears at multiple sizes."""
    lines = []
    idx = 0

    # Individual characters
    for ch in CHARS.strip():
        for font in random.sample(list(fonts.values()), min(5, len(fonts))):
            img = render_sample(ch, font)
            fname = f"mod_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{ch}")
            idx += 1

    # Every modifier at multiple sizes
    for mod in MODIFIERS:
        for font in random.sample(list(fonts.values()), min(3, len(fonts))):
            img = render_sample(mod, font)
            fname = f"mod_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{mod}")
            idx += 1

    return lines, idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate modifier training data (Futura font)")
    parser.add_argument("--font", type=str, required=True,
                        help="Path to Futura .ttf file")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=Path,
                        default=Path("training_data/modifiers"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    fonts = {sz: ImageFont.truetype(args.font, sz) for sz in FONT_SIZES}

    # Charset coverage first
    print("Generating charset coverage samples...")
    train_lines, start_idx = generate_charset_coverage(fonts, args.output)
    print(f"  {len(train_lines)} charset coverage samples")

    # Main generation
    print(f"Generating {args.count} random samples...")
    for i in range(args.count):
        text = random_text()
        font = fonts[random.choice(FONT_SIZES)]
        img = render_sample(text, font)
        fname = f"mod_{start_idx + i:06d}.png"
        img.save(args.output / fname)
        train_lines.append(f"{fname}\t{text}")

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{args.count}")

    # Write char dict
    chars = sorted(set(CHARS.strip()))
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
    print(f"  Dict: {dict_path} ({len(chars)} chars)")


if __name__ == "__main__":
    main()
