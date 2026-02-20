# OverwatchLooker

Overwatch 2 match analyzer powered by Claude Vision. Captures your Tab screen and extracts match data (map, mode, scores, player stats, hero-specific abilities) into structured text.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with your credentials:

```env
# Option 1: Direct Anthropic API
ANTHROPIC_API_KEY=sk-ant-...

# Option 2: Amazon Bedrock
AWS_KEY=AKIA...
AWS_SECRET=...
AWS_REGION=eu-west-1
```

## Usage

### Hotkey mode (default)

```bash
uv run python main.py
```

Runs as a system tray application. Hold **Tab** and click a **mouse side button** to capture and analyze the current screen.

- Tray icon appears in the system tray (right-click for Start/Stop/Quit)
- Analysis prints to the console and copies to clipboard
- A notification pops up on the second monitor when done
- Press **Ctrl+C** in the terminal to quit

### Image file mode

```bash
uv run python main.py screenshot.png
```

Analyzes a saved screenshot directly without hotkey listening.

## Output

```
============================================================
  OVERWATCH LOOKER -- Analysis at 2026-02-20 06:19:57
============================================================
MAP: Lijiang Tower
TIME: 8:06
MODE: Control
RESULT: DEFEAT

=== YOUR TEAM ===
Role | Player | E | A | D | DMG | H | MIT
TANK | RUUKOYU | 13 | 0 | 4 | 8,990 | 2,037 | 10,102
...

HERO STATS:
Mizuki - Players Saved: 11; Weapon Accuracy: 31%, ...
============================================================
```

## Configuration

Edit `overwatchlooker/config.py` to change:

- `ANTHROPIC_MODEL` / `BEDROCK_MODEL` -- which Claude model to use
- `MONITOR_INDEX` -- which monitor to screenshot (1 = primary)
- `MAX_TOKENS` -- max response tokens
