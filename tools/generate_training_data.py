"""Generate synthetic training data for Tesseract fine-tuning on OW2 scoreboard font.

Renders text lines in Big Noodle Too Oblique on OW2-style backgrounds,
outputting paired .png + .gt.txt files for tesstrain.

Usage:
    uv run python tools/generate_training_data.py [--output DIR] [--count N]
"""

import argparse
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FONTS_DIR = Path(__file__).parent.parent / "fonts"
_FONT_OBLIQUE = str(_FONTS_DIR / "BigNoodleTooOblique.ttf")
_FONT_REGULAR = str(_FONTS_DIR / "BigNoodleToo.ttf")

# Characters the model needs to recognize
# Big Noodle Too is a titling font — lowercase renders identical to uppercase.
# All ground truth must be UPPERCASE to avoid ambiguity.
LATIN_UPPER = string.ascii_uppercase
DIGITS = string.digits
CYRILLIC_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
UKRAINIAN_UPPER = "ІЇЄҐ"
PUNCTUATION = ".,:-()/%'\"!? "
ACCENTED_UPPER = "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"

ALL_CHARS = (LATIN_UPPER + DIGITS + CYRILLIC_UPPER +
             UKRAINIAN_UPPER + PUNCTUATION + ACCENTED_UPPER)

# OW2-style background colors (RGB)
_BG_COLORS = [
    (20, 40, 60),    # dark blue (ally team)
    (25, 50, 70),    # lighter ally blue
    (30, 55, 80),    # ally row highlight
    (1, 78, 108),    # actual ally bg from screenshot
    (1, 126, 171),   # self-row highlight
    (3, 65, 91),     # stat column
    (2, 107, 146),   # self stat column
    (60, 25, 30),    # dark red (enemy team)
    (70, 30, 35),    # lighter enemy red
    (80, 35, 40),    # enemy row highlight
    (15, 20, 30),    # dark panel background
    (25, 30, 40),    # hero stats panel
    (10, 10, 15),    # very dark (header area)
    (40, 60, 80),    # teal accent
]

# Text colors used in-game
_TEXT_COLORS = [
    (255, 255, 255),  # white (primary)
    (255, 255, 255),  # white again (most common)
    (255, 255, 255),  # white again
    (220, 220, 220),  # light gray
    (200, 200, 200),  # gray
    (100, 200, 255),  # cyan/blue (ally highlight)
    (255, 200, 100),  # gold/yellow (numbers)
    (180, 180, 180),  # muted (titles)
]

# --- Text generators ---

_NAME_PARTS = [
    "SHADOW", "DARK", "LIGHT", "FIRE", "ICE", "STORM", "WOLF", "DRAGON",
    "NOVA", "STAR", "MOON", "SUN", "CYBER", "MEGA", "ULTRA", "HYPER",
    "NINJA", "SAMURAI", "KNIGHT", "QUEEN", "KING", "LORD", "FURY",
    "HAWK", "EAGLE", "VIPER", "COBRA", "PHOENIX", "TITAN", "GHOST",
    "BLAZE", "FROST", "THUNDER", "BOLT", "FLASH", "SWIFT", "IRON",
    "STEEL", "SILVER", "GOLDEN", "RUBY", "JADE", "ONYX", "CRYSTAL",
    "RAGE", "BLADE", "ALPHA", "OMEGA", "DELTA", "SIGMA", "ZERO",
    "NEO", "MAX", "ACE", "REX", "ZAP", "DOC", "VEX", "RAY", "GEL",
]

_CYRILLIC_NAMES = [
    "СЕРОУХИЙАЗАР", "ЛУНАВОД", "КИБЕРВОЙН", "ТЕНЕВОЙДУХ", "ОГНЕБОРЕЦ",
    "ЛЕДЯНОЙВОЛК", "ШТОРМОВОЙ", "ДРАКОН", "ТАЙФУН", "НЕТРОНЬ",
    "ЗВЕЗДА", "МОЛНИЯ", "ПРИЗРАК", "ВИТЯЗЬ", "СНАЙПЕР", "СТРАЖ",
    "ФЕНИКС", "ТИТАН", "РЫЦАРЬ", "БЕРСЕРК", "ВАЛЬКИРИЯ", "КОСМОНАВТ",
    "ВОЛК", "МЕДВЕДЬ", "СОКОЛ", "ВОИН", "МАРШАЛ", "ВИХРЬ", "ГРОМ",
]

_HERO_NAMES = [
    "REINHARDT", "D.VA", "WINSTON", "ZARYA", "SIGMA", "ROADHOG",
    "JUNKER QUEEN", "RAMATTRA", "ORISA", "WRECKING BALL", "DOOMFIST",
    "GENJI", "TRACER", "WIDOWMAKER", "HANZO", "CASSIDY", "ASHE",
    "REAPER", "SOLDIER: 76", "SOJOURN", "ECHO", "PHARAH", "BASTION",
    "JUNKRAT", "MEI", "TORBJORN", "SOMBRA", "SYMMETRA",
    "ANA", "MERCY", "LUCIO", "MOIRA", "KIRIKO", "ZENYATTA",
    "JUNO", "MIZUKI", "LIFEWEAVER", "ILLARI", "VENTURE", "EMRE",
    "ANRAN", "VENDETTA",
]

_MAP_NAMES = [
    "LIJIANG TOWER", "NUMBANI", "KING'S ROW", "HAVANA", "COLOSSEO",
    "NEW JUNK CITY", "SAMOA", "RUNASAPI", "BLIZZARD WORLD", "ROUTE 66",
    "WATCHPOINT: GIBRALTAR", "DORADO", "HOLLYWOOD", "EICHENWALDE",
    "MIDTOWN", "PARAISO", "CIRCUIT ROYAL", "SHAMBALI MONASTERY",
    "SURAVASA", "THRONE OF ANUBIS", "HANAOKA",
]

_STAT_LABELS = [
    "CHARGE KILLS", "FIRE STRIKE KILLS", "FIRE STRIKE ACCURACY",
    "KNOCKBACK KILLS", "EARTHSHATTER STUNS", "EARTHSHATTER KILLS",
    "EARTHSHATTER DIRECT HITS", "WEAPON ACCURACY", "PLAYERS SAVED",
    "PULSAR TORPEDOES DAMAGE", "PULSAR TORPEDOES HEALING",
    "ORBITAL RAY HEALING", "DAMAGE AMPLIFIED", "ORBITAL RAY ASSISTS",
    "OBJ CONTEST TIME", "SCOPED ACCURACY", "CRITICAL HIT ACCURACY",
    "SELF HEALING", "BARRIER DAMAGE", "TELEPORTER UPTIME",
    "SOUND BARRIERS PROVIDED", "DEFENSIVE ASSISTS", "OFFENSIVE ASSISTS",
]

_TITLES = [
    "STRIKE COMMANDER", "TRUE FRIEND", "LUPINE IMPERATOR", "ASSASSIN",
    "VENDETTA'S VANGUARD", "CELESTIAL ANGEL", "YORHA SOLDIER",
    "HEARTBREAKER", "FIRECRACKER", "GINGERBREAD FRIEND", "MEDIC",
    "ORACLE", "PHANTOM THIEF", "CABBAGE MERCHANT", "ALL-STAR",
    "FEMME FATALE", "JUSTICE EXECUTION", "ALLIANCE CHAMPION",
    "STALWART HERO",
]

_MODES = ["CONTROL", "PUSH", "ESCORT", "HYBRID", "CLASH", "FLASHPOINT"]

_REAL_NAMES: list[str] = []
_REAL_NAMES_PATH = Path(__file__).parent / "mcp_player_names.txt"
if _REAL_NAMES_PATH.exists():
    _REAL_NAMES = [n.strip().upper() for n in _REAL_NAMES_PATH.read_text("utf-8").splitlines() if n.strip()]


# ---------------------------------------------------------------------------
# Text generators
# ---------------------------------------------------------------------------

def random_battletag() -> str:
    """Generate a random BattleTag-like name."""
    if _REAL_NAMES and random.random() < 0.4:
        return random.choice(_REAL_NAMES)

    r = random.random()
    if r < 0.10:
        return random.choice(_CYRILLIC_NAMES)
    elif r < 0.25:
        # Name with embedded digits (MC1R, A4B, PLAYER123)
        name = random.choice(_NAME_PARTS)
        pos = random.randint(0, len(name))
        digits = str(random.randint(1, 999))
        return name[:pos] + digits + name[pos:]
    elif r < 0.35:
        return random.choice(_NAME_PARTS) + random.choice(_NAME_PARTS)
    elif r < 0.45:
        # Short 2-4 char name (DOV, MC1R, ACE)
        length = random.randint(2, 4)
        chars = LATIN_UPPER + DIGITS
        return "".join(random.choice(chars) for _ in range(length))
    elif r < 0.55:
        base = random.choice(_NAME_PARTS)
        replacements = {"A": "Á", "E": "É", "I": "Í", "O": "Ö", "U": "Ü"}
        for old, new in replacements.items():
            if old in base and random.random() < 0.5:
                base = base.replace(old, new, 1)
                break
        return base
    else:
        return random.choice(_NAME_PARTS)


def random_time() -> str:
    """Generate a time in M:SS or MM:SS format."""
    if random.random() < 0.6:
        m = random.randint(0, 9)
        s = random.randint(0, 59)
        return f"{m}:{s:02d}"
    else:
        m = random.randint(10, 20)
        s = random.randint(0, 59)
        return f"{m}:{s:02d}"


def random_comma_number() -> str:
    """Generate a comma-separated number like 1,234 or 12,500."""
    n = random.randint(100, 99999)
    return f"{n:,}"


def random_stat_number() -> str:
    """Generate a random stat number like the scoreboard shows."""
    r = random.random()
    if r < 0.15:
        return str(random.randint(0, 30))
    elif r < 0.35:
        return random_comma_number()
    elif r < 0.50:
        return f"{random.randint(0, 100)}%"
    elif r < 0.70:
        return random_time()
    elif r < 0.85:
        return str(random.randint(0, 999))
    else:
        # Plain large number without commas
        return str(random.randint(100, 99999))


def generate_line() -> str:
    """Generate a random training line matching OW2 scoreboard text."""
    category = random.choices([
        "player_name",
        "player_row",
        "stat_numbers",
        "hero_stat",
        "header",
        "map_mode",
        "title",
        "hero_name",
        "mixed",
        "pure_chars",
        "pure_numbers",
        "pure_cyrillic",
        "time_only",
        "time_with_text",
        "comma_numbers",
        "digit_heavy",
        "digit_letter_mix",
        "apostrophe_colon",
        "random_sequence",
        "single_chars",
        "confusable_pairs",
    ], weights=[
        8,   # player_name
        8,   # player_row
        6,   # stat_numbers
        5,   # hero_stat
        2,   # header
        3,   # map_mode
        3,   # title
        3,   # hero_name
        4,   # mixed
        3,   # pure_chars
        4,   # pure_numbers
        2,   # pure_cyrillic
        5,   # time_only
        6,   # time_with_text
        6,   # comma_numbers
        8,   # digit_heavy
        8,   # digit_letter_mix
        5,   # apostrophe_colon
        5,   # random_sequence
        3,   # single_chars
        6,   # confusable_pairs
    ])[0]

    if category == "player_name":
        return random_battletag()

    elif category == "player_row":
        name = random_battletag()
        stats = "  ".join(random_stat_number() for _ in range(random.randint(4, 7)))
        return f"{name}  {stats}"

    elif category == "stat_numbers":
        return "  ".join(random_stat_number() for _ in range(random.randint(3, 8)))

    elif category == "hero_stat":
        label = random.choice(_STAT_LABELS)
        value = random_stat_number()
        return f"{label}: {value}"

    elif category == "header":
        return random.choice(["E  A  D  DMG  H  MIT", "HERO INFO  SCOREBOARD",
                              "PLAYERS SAVED", "WEAPON ACCURACY", "VS",
                              "CANCEL", "TIME:", "COMPETITIVE", "QUICKPLAY"])

    elif category == "map_mode":
        m = random.choice(_MAP_NAMES)
        mode = random.choice(_MODES)
        return random.choice([
            f"{mode} | {m}",
            f"TIME: {random_time()}",
            f"{m}  {mode}",
            f"{mode} | {m}  TIME: {random_time()}",
        ])

    elif category == "title":
        return random.choice(_TITLES)

    elif category == "hero_name":
        return random.choice(_HERO_NAMES)

    elif category == "mixed":
        parts = [random_battletag(), random.choice(_HERO_NAMES),
                 random_stat_number()]
        random.shuffle(parts)
        return "  ".join(parts)

    elif category == "pure_chars":
        chars = LATIN_UPPER + CYRILLIC_UPPER
        length = random.randint(3, 20)
        return "".join(random.choice(chars) for _ in range(length))

    elif category == "pure_numbers":
        return "  ".join(str(random.randint(0, 99999)) for _ in range(random.randint(2, 8)))

    elif category == "pure_cyrillic":
        length = random.randint(3, 15)
        return "".join(random.choice(CYRILLIC_UPPER + UKRAINIAN_UPPER)
                       for _ in range(length))

    elif category == "time_only":
        return random_time()

    elif category == "time_with_text":
        return random.choice([
            f"TIME: {random_time()}",
            f"{random_time()}  {random_comma_number()}",
            f"{random_battletag()}  {random_time()}  {random_comma_number()}",
            f"{random_comma_number()}  {random_time()}  {random.randint(0, 30)}",
            f"OBJ CONTEST TIME: {random_time()}",
            f"{random.randint(0, 30)}  {random.randint(0, 30)}  {random.randint(0, 30)}  {random_comma_number()}  {random_time()}",
            f"{random_time()}  {random_time()}  {random_comma_number()}",
            f"00:{random.randint(0,59):02d}",
        ])

    elif category == "comma_numbers":
        count = random.randint(2, 6)
        return "  ".join(random_comma_number() for _ in range(count))

    elif category == "digit_heavy":
        parts = []
        for _ in range(random.randint(3, 8)):
            r = random.random()
            if r < 0.2:
                parts.append(str(random.randint(0, 30)))
            elif r < 0.4:
                parts.append(random_comma_number())
            elif r < 0.6:
                parts.append(random_time())
            elif r < 0.8:
                parts.append(f"{random.randint(0, 100)}%")
            else:
                parts.append(str(random.randint(0, 999)))
        return "  ".join(parts)

    elif category == "digit_letter_mix":
        # Names with digits embedded — the MC1R problem
        patterns = [
            lambda: f"MC{random.randint(0,9)}R",
            lambda: f"A{random.randint(0,99)}B",
            lambda: f"X{random.randint(0,9)}Y{random.randint(0,9)}Z",
            lambda: f"{random.choice(LATIN_UPPER)}{random.randint(1,999)}{random.choice(LATIN_UPPER)}",
            lambda: f"{''.join(random.choice(LATIN_UPPER) for _ in range(2))}{random.randint(1,99)}",
            lambda: f"{random.randint(1,99)}{''.join(random.choice(LATIN_UPPER) for _ in range(2))}",
            lambda: f"{''.join(random.choice(LATIN_UPPER+DIGITS) for _ in range(random.randint(3,8)))}",
            lambda: f"PLAYER{random.randint(1,999)}",
            lambda: f"{random.choice(_NAME_PARTS)}{random.randint(0,9)}{random.randint(0,9)}",
            lambda: f"1{random.choice(LATIN_UPPER)}1{random.choice(LATIN_UPPER)}",
            lambda: f"0{random.choice(LATIN_UPPER)}0",
            lambda: f"{random.choice(LATIN_UPPER)}1{random.choice(LATIN_UPPER)}",
        ]
        return random.choice(patterns)()

    elif category == "apostrophe_colon":
        # Focus on ' and : which are easily confused
        templates = [
            "KING'S ROW",
            "VENDETTA'S VANGUARD",
            "SOLDIER: 76",
            "WATCHPOINT: GIBRALTAR",
            f"IT'S {random.choice(_NAME_PARTS)}'S TIME",
            f"TIME: {random_time()}",
            f"OBJ CONTEST TIME: {random_time()}",
            f"{random.choice(_STAT_LABELS)}: {random_stat_number()}",
            f"'{random.choice(_NAME_PARTS)}'",
            f"{random_time()}: {random_comma_number()}",
            "PLAYER'S CHOICE",
            "D'ARTAGNAN",
            "O'MALLEY",
            "MCDONALD'S",
        ]
        return random.choice(templates)

    elif category == "random_sequence":
        # Completely random character sequences for robustness
        chars = LATIN_UPPER + DIGITS + " .,:-'%"
        length = random.randint(3, 25)
        return "".join(random.choice(chars) for _ in range(length))

    elif category == "single_chars":
        # Individual characters and very short strings (for precise glyph learning)
        r = random.random()
        if r < 0.3:
            return random.choice(LATIN_UPPER + DIGITS)
        elif r < 0.6:
            return random.choice(LATIN_UPPER) + random.choice(DIGITS)
        else:
            return "".join(random.choice(LATIN_UPPER + DIGITS) for _ in range(random.randint(2, 4)))

    elif category == "confusable_pairs":
        # Pairs/sequences of commonly confused characters
        confusables = [
            "0O", "O0", "0OO0", "O0O0", "1I", "I1", "1IL", "LI1",
            "1L1L", "RА", "DA", "DB", "DО", "DO0", "1IR",
            "S5", "5S", "S5S5", "8B", "B8", "6G", "G6",
            "2Z", "Z2", "CG", "UV", "VU", "WM", "MW",
            "00:00", "01:10", "11:11", "10:01",
            "1,111", "11,111", "10,001", "1,001",
            # With surrounding context
            "MC1R", "MC0R", "MC1A", "A1B2C3", "X0Y0Z0",
            "10,008", "10,000", "100,000",
            "0:00", "0:01", "1:00", "1:01",
        ]
        return random.choice(confusables)

    return random_battletag()


def generate_charset_lines() -> list[str]:
    """Generate lines that ensure every target character appears at least once."""
    lines = []
    chars = list(ALL_CHARS)
    random.shuffle(chars)
    i = 0
    while i < len(chars):
        chunk_size = random.randint(10, 20)
        chunk = chars[i:i + chunk_size]
        lines.append("".join(chunk))
        i += chunk_size
    lines.append("!? () .,:-/ %'\"")
    lines.append(ACCENTED_UPPER)
    lines.append(UKRAINIAN_UPPER + " " + CYRILLIC_UPPER[:15])
    # Add confusable sequences to charset coverage
    lines.append("0O1IL 0:00 1:11 1,111")
    lines.append("KING'S ROW SOLDIER: 76")
    return lines


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _jitter_color(color: tuple, amount: int = 15) -> tuple:
    """Randomly shift each RGB channel."""
    return tuple(max(0, min(255, c + random.randint(-amount, amount))) for c in color)


def _add_noise(img: Image.Image, intensity: float = 0.1) -> Image.Image:
    """Add random pixel noise to simulate scoreboard transparency/game scene bleed."""
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-int(intensity * 255), int(intensity * 255) + 1,
                              arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _add_gradient(img: Image.Image) -> Image.Image:
    """Add a subtle horizontal or vertical brightness gradient."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if random.random() < 0.5:
        grad = np.linspace(random.uniform(0.85, 1.0), random.uniform(1.0, 1.15), w)
        arr *= grad[np.newaxis, :, np.newaxis]
    else:
        grad = np.linspace(random.uniform(0.9, 1.0), random.uniform(1.0, 1.1), h)
        arr *= grad[:, np.newaxis, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_line(text: str, font_path: str, font_size: int,
                bg_color: tuple, text_color: tuple,
                blur: float = 0.0) -> Image.Image:
    """Render a single text line as an image."""
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x = random.randint(4, 12)
    pad_y = random.randint(4, 8)
    img_w = text_w + pad_x * 2
    img_h = text_h + pad_y * 2

    bg = _jitter_color(bg_color, amount=random.randint(5, 20))
    img = Image.new("RGB", (img_w, img_h), bg)

    if random.random() < 0.4:
        img = _add_gradient(img)

    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=text_color, font=font)

    # Less noise at small sizes — real screenshots are actually quite clean
    if font_size <= 22:
        if random.random() < 0.3:
            img = _add_noise(img, intensity=random.uniform(0.01, 0.05))
    else:
        if random.random() < 0.6:
            img = _add_noise(img, intensity=random.uniform(0.02, 0.12))

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    return img


def generate_sample(output_dir: Path, index: int) -> None:
    """Generate one training sample (.png + .gt.txt)."""
    text = generate_line().upper()
    if not text.strip():
        text = random_battletag().upper()

    font_path = random.choice([_FONT_OBLIQUE, _FONT_OBLIQUE, _FONT_OBLIQUE,
                                _FONT_REGULAR])  # 75% oblique
    # 1080p: 18-28px, 4K: 36-56px. Heavy weight on small sizes.
    font_size = random.choices(
        [18, 19, 20, 22, 24, 26, 28, 32, 36, 40, 44, 48, 52, 56],
        weights=[12, 8, 10, 8, 8, 6, 6, 5, 5, 4, 4, 4, 3, 3],
    )[0]
    bg_color = random.choice(_BG_COLORS)
    text_color = random.choice(_TEXT_COLORS)
    # Less blur at small sizes
    if font_size <= 22:
        blur = random.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.3])
    else:
        blur = random.choice([0.0, 0.0, 0.0, 0.3, 0.5, 0.7])

    img = render_line(text, font_path, font_size, bg_color, text_color, blur)

    name = f"ow2_{index:06d}"
    img.save(output_dir / f"{name}.png")
    (output_dir / f"{name}.gt.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate OW2 Tesseract training data")
    parser.add_argument("--output", type=Path, default=Path("training_data/ow2-ground-truth"),
                        help="Output directory for .png + .gt.txt pairs")
    parser.add_argument("--count", type=int, default=69000,
                        help="Number of training samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    # Charset coverage (repeat at multiple sizes)
    coverage_lines = generate_charset_lines()
    idx = 0
    for repeat in range(10):
        for text in coverage_lines:
            font_path = random.choice([_FONT_OBLIQUE, _FONT_REGULAR])
            font_size = random.choice([18, 20, 22, 24, 28, 36, 48])
            bg_color = random.choice(_BG_COLORS)
            text_color = random.choice(_TEXT_COLORS)
            img = render_line(text, font_path, font_size, bg_color, text_color)
            name = f"ow2_{idx:06d}"
            img.save(args.output / f"{name}.png")
            (args.output / f"{name}.gt.txt").write_text(text, encoding="utf-8")
            idx += 1
    print(f"  {idx} charset coverage samples")

    print(f"Generating {args.count} random samples to {args.output}...")
    for i in range(args.count):
        generate_sample(args.output, idx + i)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{args.count}")

    # Verify character coverage
    all_text = ""
    for f in args.output.glob("*.gt.txt"):
        all_text += f.read_text(encoding="utf-8")

    covered = set(all_text)
    expected = set(ALL_CHARS)
    missing = expected - covered
    if missing:
        print(f"WARNING: {len(missing)} chars missing from training data: {''.join(sorted(missing))}")
    else:
        print(f"All {len(expected)} target chars covered.")

    print(f"Done. {idx + args.count} total samples in {args.output}")


if __name__ == "__main__":
    main()
