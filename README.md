# OverwatchLooker

Automated Overwatch 2 match tracker. Collects structured match data (map, mode, queue type, result, full roster with battletags/heroes/roles/stats, hero switches, player joins/leaves) during gameplay and prints a summary at match end.

Primary data source is **Overwolf GEP** (via OverwatchListener) which provides real-time roster, stats, hero swaps, and match lifecycle events for all 10 players. Fallback detection via **subtitle OCR** (Tesseract) for victory/defeat and hero switches when Overwolf is not available. Screen capture uses **memoir-capture** (Windows Graphics Capture + NVENC H.265). Supports **recording** gameplay sessions and **replaying** them for offline analysis. Legacy LLM analyzer backends (ChatGPT/Codex, Claude Vision) are preserved for single-image analysis but no longer used in live mode.

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
uv run python main.py --overwolf   # recommended: full Overwolf GEP data
uv run python main.py              # fallback: subtitle/chat OCR only
uv run python main.py --transcript # log subtitle OCR to transcripts/
```

Starts a system tray application with a tick-based frame loop:

1. **Overwolf GEP** provides real-time match data: roster (all 10 players with battletags, heroes, roles, stats), match lifecycle, map, mode, result
2. **Hold Tab** in-game to open the scoreboard — a screenshot is captured automatically via memoir-capture
3. **Subtitle OCR** provides fallback hero switch tracking and victory/defeat detection when Overwolf is not available
4. At match end, a **formatted summary** is printed to console and logged
5. A **notification** appears with the match result

Tray menu (right-click icon):
- **Start/Stop Listening** -- toggle hotkey + detection
- **Start/Stop Recording** -- record gameplay to `recordings/` for later replay
- **Submit last tab (win/loss)** -- manually set match result and trigger match end summary
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
uv run python main.py --replay recordings/2026-03-07_21-00-39/recording.mp4  # direct .mp4 path
uv run python main.py --replay recordings/2026-03-07_21-00-39 --no-analysis  # skip LLM
```

Replays a previously recorded session at max speed, running the full detection and analysis pipeline as if it were live. Accepts a recording directory or a direct `.mp4` file path. Useful for testing and debugging.

### CLI flags

| Flag | Description |
|---------|-------------|
| `--analyzer` | Override analyzer backend (`anthropic` or `codex`) |
| `--tg` | Send results to Telegram instead of clipboard |
| `--mcp` | Upload structured match data to the MCP server |
| `--clean` | Bypass disk cache and re-analyze from scratch |
| `--backfill` | Mark match as backfilled when uploading to MCP |
| `--transcript` | Log subtitle OCR results to `transcripts/` folder |
| `--replay` | Replay a recording directory or `.mp4` file |
| `--no-analysis` | Skip LLM analysis on detection (useful for testing replays) |
| `--win` | Hint that the match result is VICTORY |
| `--loss` | Hint that the match result is DEFEAT |
| `--ws` | Start WebSocket server for companion apps (see [protocol docs](docs/websocket-protocol.md)) |
| `--overwolf` | Start Overwolf GEP receiver (accepts OverwatchListener connections on port 28025) |

## Output format

```
═══ MATCH COMPLETE ═══
Map: Havana | Escort | Unranked | Role Queue
Result: VICTORY | Duration: 9:30
Rounds: 1 (9:30)

── ALLY (Team 1) ──
  TANK  lunavod#21722        Reinhardt -> Ramattra (1:11)    24/5/2   9483 dmg  441 heal  4732 mit
  DAMAGE bexest#2955          Pharah                          13/8/10  4752 dmg  0 heal  0 mit
  DAMAGE Szaths#2663          Emre                            18/6/0   5946 dmg  651 heal  0 mit
  SUPPORT Cucciolo#11914       Lucio -> Kiriko (1:26)          10/5/19  3544 dmg  7024 heal  0 mit
  SUPPORT YoungdaggerD#2341    Brigitte -> Jetpack Cat -> Ana  17/6/13  4603 dmg  3692 heal  814 mit

── ENEMY (Team 0) ──
  TANK  BioL2004#2426        Winston                         10/10/2  6204 dmg  1041 heal  9720 mit
  DAMAGE LeirbaXx#2447        Cassidy                         16/7/3   7620 dmg  0 heal  218 mit
  DAMAGE Skoupayou#1964       Hanzo -> Junkrat -> Ashe        12/8/2   6502 dmg  0 heal  0 mit
  SUPPORT Khraym#2826          Illari                          12/6/7   4057 dmg  6708 heal  0 mit
  SUPPORT Verdauga#2454        Ana                             6/8/5    2829 dmg  3408 heal  855 mit
═══════════════════════
```

## Detection

### Overwolf GEP (primary)

With `--overwolf`, connects to the OverwatchListener Overwolf app via WebSocket. Provides structured real-time data:

- **Match lifecycle** — start, end, rounds, outcome (VICTORY/DEFEAT)
- **Full roster** — all 10 players with battletags, heroes, roles, team assignment
- **Live stats** — K/D/A/damage/healing/mitigation updated continuously
- **Hero swaps** — detected via roster hero_name changes, with stats snapshot at swap time
- **Match metadata** — map, mode, game type, queue type, pseudo match ID

Hero names from Overwolf are fuzzy-matched against the canonical hero list for proper casing (e.g. `JETPACKCAT` → `Jetpack Cat`).

### Subtitle OCR (fallback)

Monitors the bottom-center of the screen for subtitle text using Tesseract (via pytesseract-api, direct C API bindings). Runs as part of the tick-based frame loop.

**Victory/defeat detection:**
1. **HSV pre-filter** -- checks for white pixels in the subtitle region (bottom 10%, center 20%)
2. **Tesseract confirmation** -- runs OCR only when white pixels are detected

**Hero switch tracking:**
- Detects hero names in subtitle text (e.g., "Player switched to Tracer")
- Fuzzy matching against the hero list using Levenshtein edit distance
- Deduped against Overwolf data when both sources are active

### Chat OCR (fallback)

Monitors the bottom-left of the screen for yellow chat text. Detects player join/leave events (`[player] joined the game`, `[player] left the game`) using Tesseract. Fuzzy dedup prevents OCR noise from creating duplicate events.

## Analyzer backends

### ChatGPT/Codex (`ANALYZER=codex`, default)

Uses the ChatGPT/Codex API with structured JSON schema output. Returns match data including map, mode, queue type, result, player stats, hero-specific stats, and competitive rank range. Costs are logged to `api_costs.jsonl`.

- Model: configurable via `CODEX_MODEL` (default: `gpt-5.3-codex`)
- `gpt-5.3-codex` is the most reliable model for reading unusual/stylized usernames — most other models (including 5.4) tend to misread them
- Optional reasoning effort via `CODEX_REASONING` (`low`, `medium`, `high`, `xhigh`)
- Sends zoomed crops of player names and rank area for better readability, plus a rank tier icon reference chart
- Supports per-hero crop images for multi-hero analysis

### Claude Vision (`ANALYZER=anthropic`)

Sends the screenshot to Claude with a structured JSON schema prompt. Same feature set as Codex, with one caveat: Claude Sonnet cannot reliably identify competitive rank tier icons, so rank tiers are disabled by default (only wide match detection). Claude Opus identifies them correctly.

- Model: configurable via `ANTHROPIC_MODEL` (default: `claude-sonnet-4-6`)
- Screenshots are downscaled to 1568px max width before sending

## Recording and replay

The app can record gameplay sessions for later replay and analysis.

**Recording:** Toggle via the tray menu. Uses memoir-capture's native NVENC H.265 encoder at 10 FPS, downscaled to 1080p. Per-frame keyboard state is stored as bitmasks in a `.meta` file. Overwolf GEP events (if `--overwolf` is active) are stored as JSONL in a `.overwolf.jsonl` file. Produces `recording.mp4` + `recording.meta` (+ `recording.overwolf.jsonl`) in timestamped directories under `recordings/`.

**Replay:** Use `--replay <dir>` or `--replay <file.mp4>` to replay a recording at max speed. The full detection and analysis pipeline runs as if it were live, making this useful for testing changes without playing a match.

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
| `OVERWOLF_PORT` | `28025` | Overwolf GEP receiver port (OverwatchListener connects here) |

## Project structure

```
main.py                          # CLI entry point
overwatchlooker/
  config.py                      # Environment config + constants
  match_state.py                 # Centralized match state (MatchState, PlayerState, formatting)
  tick.py                        # Tick-based frame loop (live + replay)
  tray.py                        # System tray app, Overwolf event dispatch, match lifecycle
  hotkey.py                      # Tab key listener (pynput, Windows foreground check)
  screenshot.py                  # OW2 Tab screen validation, Tesseract OCR, screenshot saving
  subtitle_listener.py           # Subtitle-based detection + hero switch tracking
  chat_listener.py               # Chat OCR for player join/leave detection
  heroes.py                      # Hero name fuzzy matching (Levenshtein)
  heroes.txt                     # Complete list of OW2 heroes
  ranks.png                      # Rank tier icon reference for analyzers
  analyzers/
    __init__.py                  # Analyzer registry
    anthropic.py                 # Claude Vision backend
    codex.py                     # ChatGPT/Codex backend
    common.py                    # Shared schema, formatting, hero merging
  recording/
    replay.py                    # MP4 + .meta replay with keyboard event synthesis
  cache.py                       # SHA256-based disk cache for analysis results
  display.py                     # Formatting, logging, safe stdout for pythonw
  notification.py                # Clipboard, tkinter overlay, audio chime
  telegram.py                    # Telethon message sending
  mcp_client.py                  # MCP server client (Streamable HTTP)
  ws_server.py                   # WebSocket server for companion apps
  overwolf.py                    # Overwolf GEP receiver (typed events, queue, recording)
docs/
  websocket-protocol.md          # WebSocket event protocol reference
recordings/                      # Recorded gameplay sessions
cache/                           # Cached analysis results
logs/                            # Timestamped log files
```
