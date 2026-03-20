"""Generate synthetic training data for hero panel label OCR.

Config-Medium font, ALL CAPS, A-Z + space only.
Varied backgrounds and text colors for robustness.

Usage:
    uv run python tools/generate_label_training.py [--count N] [--output DIR]
"""

import argparse
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = r"C:\Users\yegor\Downloads\config\Config-Medium.ttf"

# Only A-Z and space
CHARS = string.ascii_uppercase + " "

# Font sizes: 34 is 4K match, vary ±30%
FONT_SIZES = [24, 26, 28, 30, 32, 34, 34, 34, 36, 38, 40, 44]

# Background colors (RGB) — dark panel variants
BG_COLORS = [
    (0, 0, 0),         # pure black (binarized)
    (0, 0, 0),         # pure black (weighted)
    (0, 0, 0),         # pure black (weighted)
    (14, 18, 32),      # actual panel dark
    (20, 25, 40),      # panel lighter
    (10, 12, 22),      # panel darker
    (25, 30, 45),      # panel variation
    (15, 15, 15),      # neutral dark gray
    (5, 8, 18),        # very dark blue
]

# Text colors — label text is gray in-game
TEXT_COLORS = [
    (255, 255, 255),   # pure white (binarized)
    (255, 255, 255),   # pure white (weighted)
    (255, 255, 255),   # pure white (weighted)
    (102, 109, 129),   # actual label gray
    (110, 115, 135),   # slightly brighter
    (90, 97, 117),     # slightly dimmer
    (120, 125, 140),   # brighter variant
    (200, 200, 200),   # light gray
    (160, 160, 170),   # medium gray
]

# ---------------------------------------------------------------------------
# Known OW2 stat labels
# ---------------------------------------------------------------------------

STAT_LABELS = [
    # General
    "WEAPON ACCURACY", "PLAYERS SAVED", "OBJ CONTEST TIME",
    "SCOPED ACCURACY", "CRITICAL HIT ACCURACY", "SELF HEALING",
    "BARRIER DAMAGE", "TELEPORTER UPTIME", "DAMAGE AMPLIFIED",
    "SOUND BARRIERS PROVIDED", "DEFENSIVE ASSISTS", "OFFENSIVE ASSISTS",
    "HERO DAMAGE DONE", "HEALING DONE", "DAMAGE MITIGATED",
    "ELIMINATIONS", "FINAL BLOWS", "SOLO KILLS", "OBJECTIVE KILLS",
    "OBJECTIVE TIME", "MULTIKILLS", "ENVIRONMENTAL KILLS",
    "DEATHS", "ASSISTS",
    # Reinhardt
    "CHARGE KILLS", "FIRE STRIKE KILLS", "FIRE STRIKE ACCURACY",
    "KNOCKBACK KILLS", "EARTHSHATTER DIRECT HITS", "EARTHSHATTER STUNS",
    "EARTHSHATTER KILLS",
    # Juno
    "PULSAR TORPEDOES DAMAGE", "PULSAR TORPEDOES HEALING",
    "ORBITAL RAY HEALING", "ORBITAL RAY ASSIST", "ORBITAL RAY ASSISTS",
    # Moira
    "SECONDARY FIRE ACCURACY", "BIOTIC ORB KILLS", "BIOTIC ORB HEALING",
    "ALLY COALESCENCE EFFICIENCY", "ENEMY COALESCENCE EFFICIENCY",
    "COALESCENCE KILLS", "BIOTIC GRASP HEALING",
    # Ana
    "NANO BOOST ASSISTS", "ENEMIES SLEPT", "BIOTIC GRENADE KILLS",
    "SCOPED CRITICAL HIT ACCURACY", "UNSCOPED ACCURACY",
    # Mercy
    "PLAYERS RESURRECTED", "CADUCEUS BLASTER KILLS",
    "CADUCEUS BLASTER ACCURACY",
    # Lucio
    "SOUND BARRIERS PROVIDED", "SONIC AMPLIFIER KILLS",
    "SONIC AMPLIFIER ACCURACY",
    # Kiriko
    "HEALING OFUDA ACCURACY", "KITSUNE RUSH ASSISTS", "KUNAI KILLS",
    # D.Va
    "MECHS CALLED", "SELF DESTRUCT KILLS", "MICRO MISSILES KILLS",
    # Winston
    "JUMP PACK KILLS", "PRIMAL RAGE KILLS", "TESLA CANNON ACCURACY",
    "PLAYERS KNOCKED BACK",
    # Zarya
    "AVERAGE ENERGY", "HIGH ENERGY KILLS", "GRAVITON SURGE KILLS",
    "PROJECTED BARRIERS APPLIED",
    # Genji
    "DRAGONBLADE KILLS", "DAMAGE REFLECTED", "SWIFT STRIKE KILLS",
    # Tracer
    "PULSE BOMB KILLS", "PULSE BOMB ATTACHED",
    # Widowmaker
    "SCOPED CRITICAL HITS", "VENOM MINE KILLS",
    # Hanzo
    "STORM ARROW KILLS", "DRAGONSTRIKE KILLS", "CRITICAL HITS",
    # Cassidy
    "DEADEYE KILLS", "FAN THE HAMMER KILLS", "MAGNETIC GRENADE KILLS",
    # Pharah
    "ROCKET DIRECT HITS", "BARRAGE KILLS", "CONCUSSIVE BLAST KILLS",
    # Soldier 76
    "HELIX ROCKET KILLS", "TACTICAL VISOR KILLS",
    # Reaper
    "DEATH BLOSSOM KILLS", "SOULS CONSUMED",
    # Sombra
    "ENEMIES HACKED", "HACK ASSISTS",
    # Bastion
    "RECON KILLS", "ASSAULT KILLS", "ARTILLERY KILLS",
    # Junkrat
    "ENEMIES TRAPPED", "RIP TIRE KILLS", "CONCUSSION MINE KILLS",
    # Mei
    "ENEMIES FROZEN", "BLIZZARD KILLS",
    # Torbjorn
    "TURRET KILLS", "OVERLOAD KILLS", "MOLTEN CORE KILLS",
    # Symmetra
    "SENTRY TURRET KILLS",
    # Sigma
    "ACCRETION KILLS", "GRAVITIC FLUX KILLS",
    # Roadhog
    "ENEMIES HOOKED", "HOOK ACCURACY", "WHOLE HOG KILLS",
    # Orisa
    "FORTIFY DAMAGE BLOCKED", "JAVELIN KILLS", "TERRA SURGE KILLS",
    # Junker Queen
    "RAMPAGE KILLS", "CARNAGE KILLS", "COMMANDING SHOUT ASSISTS",
    # Ramattra
    "PUMMEL KILLS", "RAVENOUS VORTEX KILLS", "ANNIHILATION KILLS",
    # Wrecking Ball
    "PILEDRIVER KILLS", "MINEFIELD KILLS", "GRAPPLING CLAW KILLS",
    # Doomfist
    "METEOR STRIKE KILLS", "ROCKET PUNCH KILLS", "SEISMIC SLAM KILLS",
    # Echo
    "STICKY BOMBS KILLS", "FOCUSING BEAM KILLS", "DUPLICATE KILLS",
    # Sojourn
    "RAILGUN KILLS", "OVERCLOCK KILLS",
    # Mauga
    "CARDIAC OVERDRIVE ASSISTS", "CAGE FIGHT KILLS",
    # Lifeweaver
    "TREE OF LIFE HEALING", "PETAL PLATFORM ASSISTS",
    "LIFE GRIP SAVES",
    # Illari
    "SOLAR RIFLE KILLS", "CAPTIVE SUN KILLS", "HEALING PYLON HEALING",
    # Hazard
    "HAZARD WALL KILLS", "SPIKE TRAP KILLS",
    # Newer heroes
    "GLARE DASH KILLS", "HYPER FLUX KILLS",
]

# Words for random combinations
WORDS = [
    "FIRE", "STRIKE", "KILLS", "DAMAGE", "HEALING", "ACCURACY", "DIRECT",
    "HITS", "STUNS", "CHARGE", "KNOCKBACK", "BARRIER", "CRITICAL", "HIT",
    "SCOPED", "WEAPON", "SELF", "ALLY", "ENEMY", "EFFICIENCY", "ASSIST",
    "ASSISTS", "ELIMINATIONS", "DEATHS", "OBJECTIVE", "TIME", "FINAL",
    "BLOWS", "SOLO", "OFFENSIVE", "DEFENSIVE", "PROVIDED", "AMPLIFIED",
    "PULSE", "BOMB", "ATTACHED", "ENEMIES", "HACKED", "FROZEN", "TRAPPED",
    "ROCKETS", "TURRET", "OVERLOAD", "MOLTEN", "CORE", "SENTRY", "FLUX",
    "GRAVITON", "SURGE", "PROJECTED", "BARRIERS", "APPLIED", "AVERAGE",
    "ENERGY", "HIGH", "JUMP", "PACK", "PRIMAL", "RAGE", "TESLA", "CANNON",
    "PLAYERS", "RESURRECTED", "SAVED", "SECONDARY", "BIOTIC", "ORB",
    "ORBITAL", "RAY", "PULSAR", "TORPEDOES", "COALESCENCE", "NANO",
    "BOOST", "GRENADE", "SLEPT", "STORM", "ARROW", "DRAGON", "BLADE",
    "SWIFT", "REFLECTED", "VENOM", "MINE", "CONCUSSIVE", "BLAST",
    "HELIX", "ROCKET", "TACTICAL", "VISOR", "DEATH", "BLOSSOM", "SOULS",
    "CONSUMED", "ARTILLERY", "RECON", "ASSAULT", "BLIZZARD", "FORTIFY",
    "BLOCKED", "JAVELIN", "TERRA", "RAMPAGE", "CARNAGE", "COMMANDING",
    "SHOUT", "PUMMEL", "RAVENOUS", "VORTEX", "ANNIHILATION", "PILEDRIVER",
    "MINEFIELD", "GRAPPLING", "CLAW", "METEOR", "SEISMIC", "SLAM",
    "STICKY", "BOMBS", "FOCUSING", "BEAM", "DUPLICATE", "RAILGUN",
    "OVERCLOCK", "CARDIAC", "OVERDRIVE", "CAGE", "FIGHT", "TREE", "LIFE",
    "PETAL", "PLATFORM", "GRIP", "SAVES", "SOLAR", "RIFLE", "CAPTIVE",
    "SUN", "PYLON", "WALL", "SPIKE", "TRAP", "GLARE", "DASH", "HYPER",
    "SOUND", "SONIC", "AMPLIFIER", "CADUCEUS", "BLASTER", "KITSUNE",
    "RUSH", "KUNAI", "OFUDA", "MECHS", "CALLED", "DESTRUCT", "MICRO",
    "MISSILES", "EARTHSHATTER",
]


def random_label() -> str:
    """Generate a random label string."""
    r = random.random()
    if r < 0.40:
        return random.choice(STAT_LABELS)
    elif r < 0.70:
        n = random.randint(1, 5)
        return " ".join(random.choice(WORDS) for _ in range(n))
    elif r < 0.85:
        n = random.randint(1, 5)
        words = []
        for _ in range(n):
            length = random.randint(3, 12)
            words.append("".join(random.choice(string.ascii_uppercase) for _ in range(length)))
        return " ".join(words)
    else:
        length = random.randint(8, 20)
        return "".join(random.choice(string.ascii_uppercase) for _ in range(length))


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
    for ch in string.ascii_uppercase:
        for font in random.sample(list(fonts.values()), min(3, len(fonts))):
            img = render_sample(ch, font)
            fname = f"label_{idx:06d}.png"
            img.save(output_dir / fname)
            lines.append(f"{fname}\t{ch}")
            idx += 1

    # Character pairs (catches kerning issues)
    for _ in range(100):
        pair = random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase)
        font = random.choice(list(fonts.values()))
        img = render_sample(pair, font)
        fname = f"label_{idx:06d}.png"
        img.save(output_dir / fname)
        lines.append(f"{fname}\t{pair}")
        idx += 1

    return lines, idx


def main():
    parser = argparse.ArgumentParser(description="Generate panel label training data")
    parser.add_argument("--count", type=int, default=69000)
    parser.add_argument("--output", type=Path, default=Path("training_data/panel_labels_v2"))
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
        text = random_label()
        if not text.strip():
            text = random.choice(STAT_LABELS)

        font = fonts[random.choice(FONT_SIZES)]
        img = render_sample(text, font)
        fname = f"label_{start_idx + i:06d}.png"
        img.save(args.output / fname)
        train_lines.append(f"{fname}\t{text}")

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{args.count}")

    # Write char dict (A-Z + space)
    dict_path = args.output / "dict.txt"
    dict_path.write_text("\n".join(list(string.ascii_uppercase) + [" "]), encoding="utf-8")

    # Split 90/10 train/val
    random.shuffle(train_lines)
    split = int(len(train_lines) * 0.9)
    (args.output / "train_list.txt").write_text("\n".join(train_lines[:split]), encoding="utf-8")
    (args.output / "val_list.txt").write_text("\n".join(train_lines[split:]), encoding="utf-8")

    print(f"\nDone: {len(train_lines)} total samples in {args.output}")
    print(f"  Train: {split}, Val: {len(train_lines) - split}")
    print(f"  Dict: {dict_path} ({27} chars)")


if __name__ == "__main__":
    main()
