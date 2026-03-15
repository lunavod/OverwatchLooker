# WebSocket Protocol

OverwatchLooker exposes a WebSocket server for companion apps (e.g. Electron dashboard/overlay). Start it with the `--ws` flag:

```
uv run python main.py --ws
```

The server listens on `ws://127.0.0.1:42685` by default. Override the port with the `WS_PORT` environment variable.

## Connection

Connect to the WebSocket URL. On connect, the server immediately sends a **state snapshot** — a JSON message containing the current accumulated state. After that, real-time events are pushed as they occur.

The companion app can send **commands** to control OverwatchLooker (see [Commands](#commands) below).

## Message format

Every message is a JSON object with a `type` field:

```json
{ "type": "<event_type>", ... }
```

## Events

### `state`

Sent on connect (as the initial snapshot) and when the app starts/stops listening.

```json
{
  "type": "state",
  "active": true,
  "analyzing": false,
  "hero_map": { "PLAYER1": "Reinhardt", "PLAYER2": "Mercy" },
  "hero_history": {},
  "hero_crops": ["Reinhardt"],
  "valid_tabs": 1,
  "recording": false,
  "last_detection": "VICTORY",
  "last_analysis": { "...match data..." }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `active` | `boolean` | Whether the app is actively listening for gameplay |
| `analyzing` | `boolean` | Whether an LLM analysis is in progress |
| `hero_map` | `object` | Current player → hero mapping from subtitle OCR |
| `hero_history` | `object` | Reserved for future use |
| `hero_crops` | `string[]` | Hero names whose panel crops have been captured |
| `valid_tabs` | `number` | Number of valid Tab screenshots stored (max 2) |
| `recording` | `boolean` | Whether screen recording is active |
| `last_detection` | `string?` | Most recent detection result (`"VICTORY"` or `"DEFEAT"`) |
| `last_analysis` | `object?` | Most recent match analysis data (see `analysis` event) |

Fields like `last_detection` and `last_analysis` only appear after their first event.

---

### `tab_capture`

A valid Overwatch 2 Tab screen was captured.

```json
{
  "type": "tab_capture",
  "filename": "2026-03-15_21-33-27_tick0042.png",
  "timestamp": 4.2,
  "count": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `string` | Saved screenshot filename |
| `timestamp` | `number` | Sim time (seconds) when captured |
| `count` | `number` | Total valid Tab screenshots stored (1 or 2) |

---

### `hero_crop`

A hero stats panel was detected and the hero name was OCR'd from the Tab screen.

```json
{
  "type": "hero_crop",
  "name": "Reinhardt"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Canonical hero name (Title Case) |

---

### `hero_switch`

Subtitle OCR detected a player is now playing a different hero.

```json
{
  "type": "hero_switch",
  "player": "PLAYER1",
  "hero": "Winston",
  "time": 120.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `player` | `string` | Player name (UPPERCASE) |
| `hero` | `string` | New hero name (Title Case) |
| `time` | `number` | Sim time (seconds) when the switch was detected |

This fires for the initial hero assignment too, not just mid-match swaps.

---

### `detection`

VICTORY or DEFEAT was detected from subtitle OCR. This fires immediately when the text is recognized, before any analysis delay.

```json
{
  "type": "detection",
  "result": "VICTORY",
  "time": 482.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `result` | `string` | `"VICTORY"` or `"DEFEAT"` |
| `time` | `number` | Sim time (seconds) of detection |

---

### `analyzing`

LLM analysis has started. Sent when the analysis thread begins processing.

```json
{
  "type": "analyzing",
  "result": "VICTORY"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `result` | `string` | Detection result being analyzed |

---

### `analysis`

LLM analysis completed. Contains the full structured match data.

```json
{
  "type": "analysis",
  "data": {
    "not_ow2_tab": false,
    "map_name": "Lijiang Tower",
    "duration": "8:06",
    "mode": "CONTROL",
    "queue_type": "COMPETITIVE",
    "result": "VICTORY",
    "rank_range": {
      "min_rank": "Gold 3",
      "max_rank": "Platinum 1",
      "is_wide": false
    },
    "players": [
      {
        "team": "ALLY",
        "role": "TANK",
        "player_name": "PLAYER1",
        "title": "Stalwart Hero",
        "eliminations": 15,
        "assists": 8,
        "deaths": 3,
        "damage": 12500,
        "healing": 0,
        "mitigation": 8900,
        "is_self": true,
        "heroes": [
          {
            "hero_name": "Reinhardt",
            "started_at": "0:00",
            "stats": [
              { "label": "Charge Kills", "value": "3", "is_featured": true }
            ]
          }
        ]
      }
    ]
  }
}
```

The `data` object follows the match schema. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `map_name` | `string` | Map name |
| `duration` | `string` | Match duration as `MM:SS` |
| `mode` | `string` | `PUSH`, `CONTROL`, `ESCORT`, `HYBRID`, `CLASH`, or `FLASHPOINT` |
| `queue_type` | `string` | `COMPETITIVE` or `QUICKPLAY` |
| `result` | `string` | `VICTORY`, `DEFEAT`, or `UNKNOWN` |
| `rank_range` | `object?` | `{ min_rank, max_rank, is_wide }` or `null` |
| `players` | `array` | 10 player objects |

Each player object:

| Field | Type | Description |
|-------|------|-------------|
| `team` | `string` | `ALLY` or `ENEMY` |
| `role` | `string` | `TANK`, `DPS`, or `SUPPORT` |
| `player_name` | `string` | BattleTag (UPPERCASE) |
| `title` | `string?` | Player title or `null` |
| `eliminations` | `number?` | Elims or `null` |
| `assists` | `number?` | Assists or `null` |
| `deaths` | `number?` | Deaths or `null` |
| `damage` | `number?` | Damage dealt or `null` |
| `healing` | `number?` | Healing done or `null` |
| `mitigation` | `number?` | Damage mitigated or `null` |
| `is_self` | `boolean` | Whether this is the local player |
| `heroes` | `array` | Hero timeline (after merge) |

Each hero in the `heroes` array:

| Field | Type | Description |
|-------|------|-------------|
| `hero_name` | `string` | Hero name (Title Case) |
| `started_at` | `string?` | When this hero was first played (`M:SS` relative to match start) |
| `stats` | `array` | Hero-specific stats from the Tab screen |

## Commands

Companion apps can send JSON messages to control OverwatchLooker. Each message must have a `command` field:

```json
{ "command": "<command_name>" }
```

The server responds with either an `ok` or `error` response:

```json
{ "type": "ok", "command": "stop_listening" }
{ "type": "error", "command": "explode", "message": "Unknown command: explode" }
```

### Available commands

| Command | Description |
|---------|-------------|
| `start_listening` | Start the tick loop and begin capturing/detecting |
| `stop_listening` | Stop the tick loop and detection |
| `toggle_recording` | Start or stop screen recording |
| `submit_win` | Manually trigger analysis with VICTORY result |
| `submit_loss` | Manually trigger analysis with DEFEAT result |
| `quit` | Shut down the OverwatchLooker process |

### Error responses

| Scenario | Response |
|----------|----------|
| Invalid JSON | `{"type": "error", "message": "Invalid JSON"}` |
| Missing `command` field | `{"type": "error", "message": "Missing 'command' field"}` |
| Unknown command | `{"type": "error", "command": "...", "message": "Unknown command: ..."}` |
| Handler failed | `{"type": "error", "command": "...", "message": "..."}` |

Commands are executed synchronously — the response is sent after the handler completes. State change events (e.g. `state` with `active: true`) are broadcast to all clients as a side effect.

---

## Event timeline

A typical match produces events in this order:

```
state        → active: true (listening started)
hero_switch  → initial hero assignments as subtitles appear
tab_capture  → player presses Tab
hero_crop    → hero panel detected in Tab screenshot
hero_switch  → mid-match hero swaps
tab_capture  → another Tab press
detection    → VICTORY/DEFEAT recognized
analyzing    → LLM analysis begins
analysis     → full match data ready
state        → analyzing: false
```

## Configuration

| Env variable | Default | Description |
|-------------|---------|-------------|
| `WS_PORT` | `42685` | WebSocket server port |
