"""Generate synthetic training data for progress bar delta OCR.

Futura-style font, 0-9 + % + - only, white on black.
Training images include chevron noise (> < arrows) that the model must
learn to ignore — labels contain only the actual delta text.

Usage:
    uv run python tools/generate_delta_training.py --font /path/to/Futura.ttf [--count N] [--output DIR]
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARS = "0123456789%+-"

# Font sizes matching progress bar text at 1080p (small)
FONT_SIZES = [16, 18, 20, 22, 24, 24, 24, 26, 28, 30, 32]


def random_delta() -> str:
    """Generate a random delta percentage string."""
    r = random.random()
    sign = random.choice(["+", "-"])
    if r < 0.50:
        # Typical delta: 5-35%
        n = random.randint(5, 35)
        return f"{sign}{n}%"
    elif r < 0.70:
        # Small delta: 1-5%
        n = random.randint(1, 5)
        return f"{sign}{n}%"
    elif r < 0.85:
        # Larger delta: 35-60%
        n = random.randint(35, 60)
        return f"{sign}{n}%"
    elif r < 0.92:
        # Just percentage without sign (for robustness)
        n = random.randint(1, 50)
        return f"{n}%"
    else:
        # Edge cases
        return random.choice([
            "+1%", "-1%", "+99%", "-99%", "+50%", "-50%",
            "+10%", "-10%", "+0%", "-0%",
        ])


def draw_chevron(draw: ImageDraw.Draw, cx: float, cy: float,
                 size: float, direction: str, fill: tuple[int, int, int]) -> None:
    """Draw a single chevron (> or <) with random distortion.

    Args:
        draw: PIL ImageDraw instance
        cx, cy: center position
        size: approximate height of the chevron
        direction: 'right' for >, 'left' for <
        fill: grayscale fill value (0-255)
    """
    half_h = size / 2
    half_w = size * random.uniform(0.25, 0.5)
    thickness = random.uniform(1.0, 3.0)

    # Base chevron points
    if direction == "right":
        # > shape: tip on right
        tip_x = cx + half_w
        top = (cx - half_w, cy - half_h)
        tip = (tip_x, cy)
        bot = (cx - half_w, cy + half_h)
    else:
        # < shape: tip on left
        tip_x = cx - half_w
        top = (cx + half_w, cy - half_h)
        tip = (tip_x, cy)
        bot = (cx + half_w, cy + half_h)

    # Random distortion on each point
    def jitter(pt, amount=None):
        if amount is None:
            amount = size * 0.15
        return (pt[0] + random.uniform(-amount, amount),
                pt[1] + random.uniform(-amount, amount))

    top = jitter(top)
    tip = jitter(tip)
    bot = jitter(bot)

    # Draw as two lines (open chevron) or filled polygon
    style = random.random()
    if style < 0.5:
        # Two lines (most common appearance)
        draw.line([top, tip], fill=fill, width=max(1, int(thickness)))
        draw.line([tip, bot], fill=fill, width=max(1, int(thickness)))
    elif style < 0.75:
        # Filled thin polygon
        # Create a thin polygon around the two lines
        normal_scale = thickness * 0.5
        dx1 = tip[0] - top[0]
        dy1 = tip[1] - top[1]
        len1 = math.sqrt(dx1**2 + dy1**2) or 1
        nx1, ny1 = -dy1/len1 * normal_scale, dx1/len1 * normal_scale

        dx2 = bot[0] - tip[0]
        dy2 = bot[1] - tip[1]
        len2 = math.sqrt(dx2**2 + dy2**2) or 1
        nx2, ny2 = -dy2/len2 * normal_scale, dx2/len2 * normal_scale

        poly = [
            (top[0] + nx1, top[1] + ny1),
            (tip[0] + nx1, tip[1] + ny1),
            (tip[0] + nx2, tip[1] + ny2),
            (bot[0] + nx2, bot[1] + ny2),
            (bot[0] - nx2, bot[1] - ny2),
            (tip[0] - nx2, tip[1] - ny2),
            (tip[0] - nx1, tip[1] - ny1),
            (top[0] - nx1, top[1] - ny1),
        ]
        draw.polygon(poly, fill=fill)
    else:
        # Filled triangle (solid chevron variant)
        draw.polygon([top, tip, bot], fill=fill)


def draw_chevrons(draw: ImageDraw.Draw, x: float, cy: float,
                  direction: str, count: int, size: float) -> float:
    """Draw a group of chevrons, return total width used."""
    spacing = size * random.uniform(0.3, 0.7)
    total_w = 0
    brightness = random.randint(180, 255)

    for i in range(count):
        cx = x + total_w + size * 0.3
        # Each chevron has slightly different size/brightness
        ch_size = size * random.uniform(0.7, 1.3)
        ch_bright = max(100, min(255, brightness + random.randint(-40, 40)))
        draw_chevron(draw, cx, cy, ch_size, direction, (ch_bright, ch_bright, ch_bright))
        total_w += spacing + size * 0.5

    return total_w


def render_sample(text: str, font: ImageFont.FreeTypeFont,
                  add_chevrons: bool = True) -> Image.Image:
    """Render white text on black, optionally with chevron noise."""
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Chevron parameters
    left_chevrons = 0
    right_chevrons = 0
    chevron_size = text_h * random.uniform(0.6, 1.0)
    left_space = 0
    right_space = 0

    if add_chevrons:
        r = random.random()
        if r < 0.30:
            # No chevrons (just the text)
            pass
        elif r < 0.55:
            # Right chevrons (positive delta: +19% >>)
            right_chevrons = random.randint(1, 3)
        elif r < 0.80:
            # Left chevrons (negative delta: <<< -27%)
            left_chevrons = random.randint(1, 3)
        else:
            # Both sides (rare but possible)
            left_chevrons = random.randint(1, 2)
            right_chevrons = random.randint(1, 2)

        left_space = left_chevrons * chevron_size * random.uniform(0.6, 1.0) if left_chevrons else 0
        right_space = right_chevrons * chevron_size * random.uniform(0.6, 1.0) if right_chevrons else 0

    pad_x = random.randint(4, 15)
    pad_y = random.randint(4, 10)
    gap = random.randint(2, 8)  # gap between chevrons and text

    img_w = int(left_space + gap * bool(left_chevrons) + text_w +
                gap * bool(right_chevrons) + right_space + pad_x * 2)
    img_h = text_h + pad_y * 2

    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Position text
    text_x = pad_x + left_space + gap * bool(left_chevrons)
    text_y = pad_y
    draw.text((text_x - bbox[0], text_y - bbox[1]),
              text, fill=(255, 255, 255), font=font)

    cy = img_h / 2

    # Draw left chevrons
    if left_chevrons:
        draw_chevrons(draw, pad_x, cy, "left", left_chevrons, chevron_size)

    # Draw right chevrons
    if right_chevrons:
        rx = text_x + text_w + gap
        draw_chevrons(draw, rx, cy, "right", right_chevrons, chevron_size)

    return img


def generate_charset_coverage(fonts, output_dir):
    """Ensure every character appears at multiple sizes."""
    lines = []
    idx = 0

    for ch in CHARS:
        for font in random.sample(list(fonts.values()), min(5, len(fonts))):
            # Individual chars: no chevrons
            img = render_sample(ch, font, add_chevrons=False)
            fname = f"delta_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{ch}")
            idx += 1

    # Hard cases with and without chevrons
    hard_cases = [
        "+1%", "-1%", "+0%", "+10%", "-10%",
        "+27%", "-27%", "+19%", "-19%", "+50%", "-50%",
        "1%", "5%", "10%", "27%", "50%", "99%",
        "+5%", "-5%", "+33%", "-33%",
    ]
    for text in hard_cases:
        for font in random.sample(list(fonts.values()), min(3, len(fonts))):
            # With chevrons
            img = render_sample(text, font, add_chevrons=True)
            fname = f"delta_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{text}")
            idx += 1
            # Without chevrons
            img = render_sample(text, font, add_chevrons=False)
            fname = f"delta_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{text}")
            idx += 1

    return lines, idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate progress bar delta training data (Futura + chevron noise)")
    parser.add_argument("--font", type=str, required=True,
                        help="Path to Futura .ttf file")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=Path,
                        default=Path("training_data/progress_delta"))
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
        text = random_delta()
        font = fonts[random.choice(FONT_SIZES)]
        img = render_sample(text, font, add_chevrons=True)
        fname = f"delta_{start_idx + i:06d}.png"
        img.save(args.output / fname)
        train_lines.append(f"{fname}\t{text}")

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{args.count}")

    # Write char dict
    chars = sorted(set(CHARS))
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
