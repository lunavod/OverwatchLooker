"""Generate synthetic training data for rank screen OCR (rank + division).

Big Noodle Titling Oblique font, A-Z + 1-5 + space, white on black.
Content is rank tier names with divisions (e.g. "GOLD 3", "GRANDMASTER 1").

Usage:
    uv run python tools/generate_rank_training.py --font /path/to/big_noodle_titling_oblique.ttf [--count N] [--output DIR]
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ12345 "

# All rank tier names as they appear on screen
RANK_TIERS = [
    "BRONZE", "SILVER", "GOLD", "PLATINUM",
    "DIAMOND", "MASTER", "GRANDMASTER", "CHAMPION",
]

DIVISIONS = ["1", "2", "3", "4", "5"]

# Font sizes: rank text is moderately large on the 1080p screen
FONT_SIZES = [36, 40, 44, 48, 52, 52, 52, 56, 60, 64]


def random_rank() -> str:
    """Generate a rank + division string."""
    return f"{random.choice(RANK_TIERS)} {random.choice(DIVISIONS)}"


def random_text() -> str:
    """Generate varied text for training robustness."""
    r = random.random()
    if r < 0.50:
        # Real rank + division (primary case)
        return random_rank()
    elif r < 0.65:
        # Tier name alone
        return random.choice(RANK_TIERS)
    elif r < 0.75:
        # Division alone
        return random.choice(DIVISIONS)
    elif r < 0.85:
        # Random uppercase word (2-12 chars)
        length = random.randint(2, 12)
        return "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(length))
    elif r < 0.92:
        # Two random words
        w1 = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(2, 8)))
        w2 = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(1, 6)))
        return f"{w1} {w2}"
    else:
        # Random word + digit
        w = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(3, 10)))
        return f"{w} {random.choice(DIVISIONS)}"


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

    # Individual characters at multiple sizes
    for ch in CHARS.strip():
        for font in random.sample(list(fonts.values()), min(5, len(fonts))):
            img = render_sample(ch, font)
            fname = f"rank_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{ch}")
            idx += 1

    # Every rank + division combination
    for tier in RANK_TIERS:
        for div in DIVISIONS:
            text = f"{tier} {div}"
            for font in random.sample(list(fonts.values()), min(3, len(fonts))):
                img = render_sample(text, font)
                fname = f"rank_{idx:06d}.png"
                img.save(output_dir / fname)
                lines.append(f"{fname}\t{text}")
                idx += 1

    return lines, idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate rank division training data (Big Noodle Oblique)")
    parser.add_argument("--font", type=str, required=True,
                        help="Path to Big Noodle Titling Oblique .ttf file")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=Path,
                        default=Path("training_data/rank_division"))
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
        fname = f"rank_{start_idx + i:06d}.png"
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
