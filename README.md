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
# Optional: player identity (improves self-player detection)
OVERWATCH_USERNAME=LUNAVOD

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

### Replay mode

```bash
uv run python main.py --replay recordings/2026-03-07_21-00-39
uv run python main.py --replay recordings/2026-03-07_21-00-39/recording.mp4
```

Replays a previously recorded session at max speed, running the full detection pipeline as if it were live. Useful for testing changes without playing a match.

### CLI flags

| Flag | Description |
|---------|-------------|
| `--overwolf` | Start Overwolf GEP receiver (accepts OverwatchListener connections on port 28025) |
| `--mcp` | Submit completed matches to the MCP server (requires `MCP_URL` in `.env`) |
| `--ws` | Start WebSocket server for companion apps (see [protocol docs](docs/websocket-protocol.md)) |
| `--transcript` | Log subtitle OCR results to `transcripts/` folder |
| `--replay` | Replay a recording directory or `.mp4` file |
| `--replay-start N` | Start replay from N seconds into the recording |
| `--replay-duration N` | Only replay N seconds of the recording (from start point) |
| `--auto-recording` | Automatically record matches (starts on match start, stops after match end) |
| `--auto-recording-tail N` | Seconds to keep recording after match ends (default: 60) |

## Output format

```
═══ MATCH COMPLETE ═══
Map: Ilios | Control | Competitive | Role Queue
Result: VICTORY | Duration: 9:30
Rank: Bronze 2 — Gold 1 (WIDE)
Bans: Mercy, Symmetra, Vendetta, Zarya
Rounds: 2 (4:12, 5:18)
Score: 0:0 -> 1:0 -> 1:1 -> 2:1

── ALLY (Team 1) ──
  TANK  lunavod#21722        Reinhardt -> Ramattra (5:30)    10/9/1   8731 dmg  76 heal  17324 mit
  DAMAGE Tiefoon#21446        Ashe                            25/5/1   12525 dmg  0 heal  0 mit
  ...

── ENEMY (Team 0) ──
  TANK  PearlOgre#22851      Roadhog -> Doomfist -> Sigma    15/7/2   8316 dmg  3464 heal  7402 mit
  ...

── Reinhardt Stats ──
★ OBJ CONTEST TIME: 00:02
  CHARGE KILL: 1
  FIRE STRIKE ACCURACY: 50%
  EARTHSHATTER STUNS: 2

── Ramattra Stats ──
★ OBJ CONTEST TIME: 00:07
  WEAPON ACCURACY: 29%
  PUMMEL ACCURACY: 41%
  ANNIHILATION EFFICIENCY: 306
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

## Hero Panel OCR

At match end, the Tab screenshot is analyzed locally using finetuned PaddleOCR models:

- **Hero stats** — per-hero stat label/value pairs (e.g., "CHARGE KILLS: 1", "WEAPON ACCURACY: 29%")
- **Rank range** — competitive rank icons detected via template matching with datamined assets
- **Wide match** — yellow icon presence indicates wide skill gap
- **Hero bans** — banned hero portraits matched against datamined hero art

See [hero-panel-ocr.md](docs/hero-panel-ocr.md) for technical details and [training-ocr-models.md](docs/training-ocr-models.md) for retraining.

## Recording and replay

The app can record gameplay sessions for later replay and analysis.

**Manual recording:** Toggle via the tray menu. Uses memoir-capture's native NVENC H.265 encoder at 10 FPS, downscaled to 1080p. Per-frame keyboard state is stored as bitmasks in a `.meta` file. Overwolf GEP events (if `--overwolf` is active) are stored as JSONL in a `.overwolf.jsonl` file. Produces `recording.mp4` + `recording.meta` (+ `recording.overwolf.jsonl`) in timestamped directories under `recordings/`.

**Auto-recording:** With `--auto-recording`, recording starts automatically on match start and stops after a configurable tail period (default 60s, set via `--auto-recording-tail`). Saved to `recordings_auto/`. A `match_info.json` file is written at match end with map, mode, result, team side, score progression, duration, and start time. Completed recordings are automatically uploaded in the background via tus resumable upload (requires `TUSD_UPLOAD_URL` and `TUSD_AUTH_KEY` in `.env`).

**Replay:** Use `--replay <dir>` or `--replay <file.mp4>` to replay a recording at max speed. The full detection and analysis pipeline runs as if it were live, making this useful for testing changes without playing a match.

## Configuration

All settings are in `overwatchlooker/config.py`, loaded from environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `OVERWATCH_USERNAME` | -- | Your BattleTag (improves self-player detection) |
| `MCP_URL` | -- | MCP server URL for match data upload |
| `MCP_SOURCE` | `"looker"` | Source identifier sent with MCP submissions |
| `WS_PORT` | `42685` | WebSocket server port for companion apps |
| `OVERWOLF_PORT` | `28025` | Overwolf GEP receiver port (OverwatchListener connects here) |
| `TUSD_UPLOAD_URL` | -- | tus upload endpoint for recording uploads |
| `TUSD_AUTH_KEY` | -- | Bearer token for tus upload authentication |

## Project structure

```
main.py                          # CLI entry point
overwatchlooker/
  config.py                      # Environment config + constants
  match_state.py                 # Centralized match state (MatchState, PlayerState, formatting)
  hero_panel.py                  # Hero panel OCR, rank detection, hero ban detection
  tick.py                        # Tick-based frame loop (live + replay)
  tray.py                        # System tray app, Overwolf event dispatch, match lifecycle
  screenshot.py                  # OW2 Tab screen validation, screenshot saving
  subtitle_listener.py           # Subtitle-based detection + hero switch tracking (fallback)
  chat_listener.py               # Chat OCR for player join/leave detection (fallback)
  heroes.py                      # Hero name fuzzy matching (Levenshtein)
  heroes.txt                     # Complete list of OW2 heroes
  recording/
    replay.py                    # MP4 + .meta replay with keyboard event synthesis
  display.py                     # Formatting, logging, safe stdout for pythonw
  notification.py                # Desktop notifications + audio chime
  mcp_client.py                  # MCP server client (Streamable HTTP)
  ws_server.py                   # WebSocket server for companion apps
  overwolf.py                    # Overwolf GEP receiver (typed events, queue, recording)
  recording_uploader.py          # Background tus resumable upload of recordings
  models/                        # Finetuned PaddleOCR inference models (Git LFS)
    panel_labels/                # Stat label OCR (Config Medium, A-Z + space)
    panel_values/                # Stat value OCR (Futura, 0-9 + % + , + . + :)
    panel_featured/              # Featured stat value OCR (Big Noodle Titling Oblique)
    team_side/                   # ATTACK/DEFEND detection OCR (Big Noodle Titling Oblique)
  assets/                        # Datamined game assets for template matching
    ranks/                       # Rank tier icons + division signs + wide match icon
    heroes/                      # Hero portrait thumbnails (50 heroes)
docs/
  websocket-protocol.md          # WebSocket event protocol reference
  hero-panel-ocr.md              # Hero panel OCR + rank detection technical reference
  training-ocr-models.md         # Guide for training new OCR models
recordings/                      # Manual recorded gameplay sessions
recordings_auto/                 # Auto-recorded match sessions
logs/                            # Timestamped log files
```
