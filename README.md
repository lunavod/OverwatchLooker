# OverwatchLooker

Automated Overwatch 2 match analyzer. Captures your Tab scoreboard, extracts structured match data (map, mode, queue type, result, player stats, hero switches), and copies it to clipboard, sends it to Telegram, or uploads it to an MCP server.

Two analyzer backends: **ChatGPT/Codex** (cloud, default) or **Claude Vision** (cloud). Victory/defeat is detected automatically via **subtitle OCR** with Tesseract. Supports **recording** gameplay sessions and **replaying** them for offline analysis.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file:

```env
# Analyzer backend: "codex" (default) or "anthropic"
ANALYZER=codex

# Required for Anthropic backend (if using ANALYZER=anthropic)
ANTHROPIC_API_KEY=sk-ant-...

# Optional: player identity (improves self-player detection)
OVERWATCH_USERNAME=LUNAVOD

# Optional: Telegram integration (Telethon user client)
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_CHANNEL=...   # chat ID (e.g. -5033067937)

# Optional: MCP server for match data storage
MCP_URL=https://your-mcp-server.example.com/mcp
```

## Usage

### Tray mode (default)

```bash
uv run python main.py
uv run python main.py --tg       # send results to Telegram
uv run python main.py --mcp      # upload structured match data to MCP server
uv run python main.py --transcript  # log subtitle OCR to transcripts/
```

Starts a system tray application with a tick-based frame loop:

1. **Hold Tab** in-game to open the scoreboard — a screenshot is captured automatically via dxcam
2. **Hero switches are tracked** in real time via subtitle OCR (Tesseract)
3. **Victory/defeat is detected automatically** via subtitle OCR
4. Analysis runs, result is **copied to clipboard** (or sent to Telegram with `--tg`)
5. A **notification** appears on the second monitor with a chime

Tray menu (right-click icon):
- **Start/Stop Listening** -- toggle hotkey + detection
- **Start/Stop Recording** -- record gameplay to `recordings/` for later replay
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

### Replay mode

```bash
uv run python main.py --replay recordings/2026-03-07_21-00-39
uv run python main.py --replay recordings/2026-03-07_21-00-39 --no-analysis  # skip LLM
uv run python main.py --replay recordings/2026-03-07_21-00-39 --no-cache     # don't cache decompressed frames
```

Replays a previously recorded session at max speed, running the full detection and analysis pipeline as if it were live. Useful for testing and debugging.

### CLI flags

| Flag | Description |
|---------|-------------|
| `--analyzer` | Override analyzer backend (`anthropic` or `codex`) |
| `--tg` | Send results to Telegram instead of clipboard |
| `--mcp` | Upload structured match data to the MCP server |
| `--clean` | Bypass disk cache and re-analyze from scratch |
| `--backfill` | Mark match as backfilled when uploading to MCP |
| `--transcript` | Log subtitle OCR results to `transcripts/` folder |
| `--replay` | Replay a recording directory instead of live capture |
| `--no-cache` | Skip decompressing frames to disk cache (slower replay) |
| `--no-analysis` | Skip LLM analysis on detection (useful for testing replays) |
| `--win` | Hint that the match result is VICTORY |
| `--loss` | Hint that the match result is DEFEAT |
| `--ws` | Start WebSocket server for companion apps (see [protocol docs](docs/websocket-protocol.md)) |

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

## Detection

### Subtitle OCR

Monitors the bottom-center of the screen for subtitle text using Tesseract (via pytesseract-api, direct C API bindings). Runs as part of the tick-based frame loop.

**Victory/defeat detection:**
1. **HSV pre-filter** -- checks for white pixels in the subtitle region (bottom 10%, center 20%)
2. **Tesseract confirmation** -- runs OCR only when white pixels are detected
3. **Configurable delay** -- waits 5 seconds after detection before triggering analysis

**Hero switch tracking:**
- Detects hero names in subtitle text (e.g., "Player switched to Tracer")
- Fuzzy matching against the hero list using Levenshtein edit distance
- Builds per-player hero history with timestamps for the full match

Only activates when `overwatch.exe` is the foreground window. 30-second cooldown between detections.

## Analyzer backends

### ChatGPT/Codex (`ANALYZER=codex`, default)

Uses the ChatGPT/Codex API with structured JSON schema output. Returns match data including map, mode, queue type, result, player stats, hero-specific stats, and competitive rank range. Costs are logged to `api_costs.jsonl`.

- Model: configurable via `CODEX_MODEL` (default: `gpt-5.3-codex`)
- `gpt-5.3-codex` is the most reliable model for reading unusual/stylized usernames — most other models (including 5.4) tend to misread them
- Optional reasoning effort via `CODEX_REASONING` (`low`, `medium`, `high`, `xhigh`)
- Sends zoomed crops of player names and rank area for better readability
- Supports per-hero crop images for multi-hero analysis

### Claude Vision (`ANALYZER=anthropic`)

Sends the screenshot to Claude with a structured JSON schema prompt. Same feature set as Codex, with one caveat: Claude Sonnet cannot reliably identify competitive rank tier icons, so rank tiers are disabled by default (only wide match detection). Claude Opus identifies them correctly.

- Model: configurable via `ANTHROPIC_MODEL` (default: `claude-sonnet-4-6`)
- Screenshots are downscaled to 1568px max width before sending

## Recording and replay

The app can record gameplay sessions for later replay and analysis.

**Recording:** Toggle via the tray menu. Records screen frames at 10 FPS with zstd compression, downscaled to 1080p. Keyboard events are logged with frame numbers for deterministic replay. Recordings are saved to `recordings/` with timestamped directories.

**Replay:** Use `--replay <dir>` to replay a recording at max speed. The full detection and analysis pipeline runs as if it were live, making this useful for testing changes without playing a match.

## MCP integration

With `--mcp`, structured match data (map, mode, players, stats, hero details, hero switch history) is uploaded to an MCP server via Streamable HTTP after each analysis. Requires `MCP_URL` in `.env`.

## Telegram integration

Uses [Telethon](https://docs.telethon.dev/) (MTProto user client, not bot API). First run requires interactive login to create a session file.

Requires `.env` variables: `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_CHANNEL`.

## Configuration

All settings are in `overwatchlooker/config.py`, loaded from environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `ANALYZER` | `"codex"` | Backend: `"codex"` or `"anthropic"` |
| `ANTHROPIC_API_KEY` | -- | Required for Anthropic backend |
| `ANTHROPIC_MODEL` | `"claude-sonnet-4-6"` | Claude model to use |
| `SONNET_RANK_TIERS` | `false` | Enable rank tier extraction on Sonnet (unreliable) |
| `CODEX_MODEL` | `"gpt-5.3-codex"` | ChatGPT/Codex model to use |
| `CODEX_REASONING` | -- | Reasoning effort: `low`, `medium`, `high`, `xhigh` |
| `MAX_TOKENS` | `16000` | Max response tokens |
| `MONITOR_INDEX` | `1` | Which monitor to capture (1 = primary) |
| `SCREENSHOT_MAX_AGE_SECONDS` | `120` | Max age of screenshot before it's considered stale |
| `SUBTITLE_POLL_INTERVAL` | `1.0` | Seconds between subtitle region checks |
| `AUDIO_COOLDOWN_SECONDS` | `30.0` | Min seconds between detections |
| `OVERWATCH_USERNAME` | -- | Your BattleTag (improves self-player detection) |
| `MCP_URL` | -- | MCP server URL for match data upload |
| `MCP_SOURCE` | `"looker"` | Source identifier sent with MCP submissions |
| `WS_PORT` | `42685` | WebSocket server port for companion apps |

## Project structure

```
main.py                          # CLI entry point
overwatchlooker/
  config.py                      # Environment config + constants
  tick.py                        # Tick-based frame loop (live + replay)
  tray.py                        # System tray app, analysis orchestration
  hotkey.py                      # Tab key listener (pynput, Windows foreground check)
  screenshot.py                  # Screen capture (dxcam), OW2 Tab validation, Tesseract OCR
  subtitle_listener.py           # Subtitle-based detection + hero switch tracking
  heroes.py                      # Hero name fuzzy matching (Levenshtein)
  heroes.txt                     # Complete list of OW2 heroes
  analyzers/
    __init__.py                  # Analyzer registry
    anthropic.py                 # Claude Vision backend
    codex.py                     # ChatGPT/Codex backend
    common.py                    # Shared schema, formatting, hero merging
  recording/
    recorder.py                  # zstd-compressed frame + keyboard recording
    replay.py                    # Frame decompression + event replay
  cache.py                       # SHA256-based disk cache for analysis results
  display.py                     # Formatting, logging, safe stdout for pythonw
  notification.py                # Clipboard, tkinter overlay, audio chime
  telegram.py                    # Telethon message sending
  mcp_client.py                  # MCP server client (Streamable HTTP)
  ws_server.py                   # WebSocket server for companion apps
docs/
  websocket-protocol.md          # WebSocket event protocol reference
recordings/                      # Recorded gameplay sessions
cache/                           # Cached analysis results
logs/                            # Timestamped log files
```
