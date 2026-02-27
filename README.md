# OverwatchLooker

Automated Overwatch 2 match analyzer. Captures your Tab scoreboard, extracts structured match data (map, mode, queue type, result, player stats, hero abilities), and copies it to clipboard or sends it to Telegram.

Two analyzer backends: **Claude Vision** (cloud, default) or **EasyOCR** (local, GPU-accelerated). Two detection modes for automatic triggering: **subtitle OCR** (default) or **audio matching**.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file:

```env
# Analyzer backend: "claude" (default) or "ocr"
ANALYZER=claude

# Required for Claude backend
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Telegram integration (Telethon user client)
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_CHANNEL=...   # chat ID (e.g. -5033067937)
```

## Usage

### Tray mode (default)

```bash
uv run python main.py
uv run python main.py --tg       # send results to Telegram
uv run python main.py --audio    # use audio detection instead of subtitle OCR
```

Starts a system tray application:

1. **Hold Tab** in-game to open the scoreboard — a screenshot is captured automatically
2. **Victory/defeat is detected automatically** via subtitle OCR (or audio matching with `--audio`)
3. Analysis runs, result is **copied to clipboard** (or sent to Telegram with `--tg`)
4. A **notification** appears on the second monitor with a chime

Tray menu (right-click icon):
- **Start/Stop Listening** -- toggle hotkey + detection
- **Submit last tab (win/loss)** -- manually trigger analysis with a known result
- **Quit**

Press **Ctrl+C** in the terminal to exit.

### Image file mode

```bash
uv run python main.py screenshot.png
uv run python main.py screenshot.png --clean    # bypass cache, re-analyze
uv run python main.py screenshot.png --win      # hint result as VICTORY
uv run python main.py screenshot.png --loss     # hint result as DEFEAT
uv run python main.py screenshot.png --tg       # send to Telegram
```

Analyzes a saved screenshot directly. Results are cached on disk (`cache/`) and reused on subsequent runs unless `--clean` is passed.

### CLI flags

| Flag | Description |
|---------|-------------|
| `--tg` | Send results to Telegram instead of clipboard |
| `--audio` | Use audio-based victory/defeat detection (requires proc-tap) |
| `--clean` | Bypass disk cache and re-analyze from scratch |
| `--win` | Hint that the match result is VICTORY |
| `--loss` | Hint that the match result is DEFEAT |

## Output format

```
============================================================
  OVERWATCH LOOKER -- Analysis at 2026-02-20 06:19:57
============================================================
MAP: Lijiang Tower
TIME: 8:06
MODE: Control
QUEUE: COMPETITIVE
RESULT: DEFEAT

=== YOUR TEAM ===
Role | Player | E | A | D | DMG | H | MIT
TANK | RUUKOYU | 13 | 0 | 4 | 8,990 | 2,037 | 10,102
DPS | Player2 | 8 | 3 | 5 | 6,200 | 0 | 0
DPS | Player3 | 10 | 2 | 3 | 7,100 | 0 | 0
SUPPORT | Player4 | 2 | 15 | 4 | 2,300 | 8,500 | 0
SUPPORT | Player5 | 3 | 12 | 6 | 1,800 | 7,200 | 0

=== ENEMY TEAM ===
Role | Player | E | A | D | DMG | H | MIT
...

HERO STATS:
Mizuki - Players Saved: 11; Weapon Accuracy: 31%; ...
============================================================
```

## Detection modes

### Subtitle OCR (default)

Monitors the bottom-center of the screen for the ATHENA subtitle text ("VICTORY" / "DEFEAT"). Uses a fast two-stage approach:

1. **HSV pre-filter** -- checks for 500+ white pixels in the subtitle region (bottom 10%, center 20%)
2. **EasyOCR confirmation** -- runs OCR only when white pixels are detected

Only activates when `overwatch.exe` is the foreground window. 30-second cooldown between detections.

### Audio matching (`--audio`)

Captures Overwatch process audio via WASAPI loopback (per-process, no system sounds) using [proc-tap](https://github.com/EricLBuehler/proc-tap). Matches against reference clips in `refs/` using normalized cross-correlation.

- Ring buffer stores last 4 seconds of audio, processed every 0.5s
- Requires 2 consecutive matching hops above threshold
- Winner must beat runner-up by a configurable margin
- 30-second cooldown between detections

Reference clips go in `refs/` as `victory.wav` and `defeat.wav` (also supports `.ogg`, `.flac`).

## Analyzer backends

### Claude Vision (`ANALYZER=claude`, default)

Sends the screenshot to Claude with a structured prompt. Returns formatted text. Costs are logged to `api_costs.jsonl`.

- Model: configurable via `ANTHROPIC_MODEL` (default: `claude-sonnet-4-6`)
- Outputs `QUEUE TYPE:` line (COMPETITIVE or QUICKPLAY)

### EasyOCR (`ANALYZER=ocr`)

Fully local analysis using EasyOCR + OpenCV. No API calls, no cost.

- GPU-accelerated (CUDA 12.4)
- Hybrid approach: batch OCR for large text + focused cell-level OCR for small stats
- HSV brightness analysis to detect OW2's dim "0" values vs. bright non-zero stats
- Region-based extraction with normalized coordinates for resolution independence
- Detects competitive vs quickplay from "- COMPETITIVE" suffix in mode header

## Telegram integration

Uses [Telethon](https://docs.telethon.dev/) (MTProto user client, not bot API). First run requires interactive login to create a session file.

Requires `.env` variables: `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_CHANNEL`.

## Configuration

All settings are in `overwatchlooker/config.py`, loaded from environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `ANALYZER` | `"claude"` | Backend: `"claude"` or `"ocr"` |
| `ANTHROPIC_API_KEY` | -- | Required for Claude backend |
| `ANTHROPIC_MODEL` | `"claude-sonnet-4-6"` | Claude model to use |
| `MAX_TOKENS` | `16000` | Max response tokens |
| `MONITOR_INDEX` | `1` | Which monitor to capture (1 = primary) |
| `SCREENSHOT_MAX_AGE_SECONDS` | `120` | Max age of screenshot before it's considered stale |
| `SUBTITLE_POLL_INTERVAL` | `1.0` | Seconds between subtitle region checks |
| `AUDIO_CHUNK_DURATION` | `4.0` | Ring buffer length (seconds) |
| `AUDIO_HOP_DURATION` | `0.5` | Processing interval (seconds) |
| `AUDIO_COOLDOWN_SECONDS` | `30.0` | Min seconds between detections |
| `AUDIO_MATCH_THRESHOLD` | `0.25` | NCC threshold for match |
| `AUDIO_CONFIRM_HOPS` | `2` | Consecutive hops required to confirm |
| `AUDIO_MATCH_MARGIN` | `0.10` | Winner must beat runner-up by this |
| `AUDIO_MIN_RMS` | `0.0005` | Minimum RMS energy to attempt matching |

## Project structure

```
main.py                          # CLI entry point
overwatchlooker/
  config.py                      # Environment config + constants
  tray.py                        # System tray app, screenshot loop, analysis orchestration
  hotkey.py                      # Tab key listener (pynput, Windows foreground check)
  screenshot.py                  # Monitor capture (mss), OW2 Tab screen validation
  ocr_analyzer.py                # EasyOCR + OpenCV scoreboard extraction
  analyzer.py                    # Claude Vision API scoreboard extraction
  subtitle_listener.py           # Subtitle-based VICTORY/DEFEAT detection
  audio_listener.py              # Audio-based VICTORY/DEFEAT detection (proc-tap)
  cache.py                       # SHA256-based disk cache for analysis results
  display.py                     # Formatting, logging, safe stdout for pythonw
  notification.py                # Clipboard, tkinter overlay, audio chime
  telegram.py                    # Telethon message sending
refs/                            # Audio reference clips (victory.wav, defeat.wav)
cache/                           # Cached analysis results
logs/                            # Timestamped log files
```
