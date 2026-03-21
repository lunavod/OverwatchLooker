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
  "hero_map": { "PLAYER1": "Reinhardt", "PLAYER2": "Mercy" },
  "hero_history": {},
  "hero_crops": ["Reinhardt"],
  "valid_tabs": 1,
  "recording": false,
  "last_detection": "VICTORY"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `active` | `boolean` | Whether the app is actively listening for gameplay |
| `hero_map` | `object` | Current player → hero mapping from subtitle OCR |
| `hero_history` | `object` | Reserved for future use |
| `hero_crops` | `string[]` | Hero names whose panel crops have been captured |
| `valid_tabs` | `number` | Number of valid Tab screenshots stored (max 2) |
| `recording` | `boolean` | Whether screen recording is active |
| `last_detection` | `string?` | Most recent detection result (`"VICTORY"` or `"DEFEAT"`) |

Fields like `last_detection` only appear after their first event.

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

### `player_change`

Chat OCR detected a player joining or leaving the game.

```json
{
  "type": "player_change",
  "player": "TERYASCOTCH",
  "event": "left",
  "time": 245.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `player` | `string` | Player name (UPPERCASE) |
| `event` | `string` | `"joined"` or `"left"` |
| `time` | `number` | Sim time (seconds) when the event was detected |

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

### `match_complete`

Match ended. Contains a summary of the collected match data.

```json
{
  "type": "match_complete",
  "data": {
    "result": "VICTORY",
    "map": "Havana",
    "mode": "Escort",
    "duration_ms": 570000
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `result` | `string?` | `"VICTORY"`, `"DEFEAT"`, or `null` if unknown |
| `map` | `string` | Map name (may be empty if Overwolf not connected) |
| `mode` | `string` | Game mode name (may be empty) |
| `duration_ms` | `number?` | Match duration in milliseconds, or `null` |
| `rank_min` | `string?` | Minimum rank in the lobby (e.g. "Bronze 2"), or `null` |
| `rank_max` | `string?` | Maximum rank in the lobby (e.g. "Gold 1"), or `null` |
| `is_wide_match` | `boolean?` | Whether the match has a wide skill range |
| `hero_bans` | `string[]?` | List of banned hero names, or `null` |

### `match_state`

Emitted whenever any part of the match state changes — Overwolf events (roster, map, mode, rounds, outcome), subtitle/chat detections, or match lifecycle transitions. Contains the full serialized match state.

```json
{
  "type": "match_state",
  "data": {
    "map": "Ilios",
    "map_code": "1645",
    "mode": "Control",
    "mode_code": "23",
    "game_type": "RANKED",
    "queue_type": "ROLE_QUEUE",
    "result": null,
    "result_source": null,
    "started_at": 1774119842381,
    "ended_at": null,
    "duration_ms": null,
    "rounds": [{"started": 1774119864335, "ended": null}],
    "pseudo_match_id": "9d59eeff-...",
    "rank_min": null,
    "rank_max": null,
    "is_wide_match": null,
    "hero_bans": null,
    "local_team": 1,
    "hero_tabs": {},
    "latest_tab": null,
    "players": {
      "LUNAVOD": {
        "battletag": "lunavod#21722",
        "team": 1,
        "team_side": "ALLY",
        "is_local": true,
        "role": "TANK",
        "slot": 9,
        "hero_swaps": [{"hero": "Reinhardt", "at": 1774119863346, "source": "overwolf:roster", "stats": null}],
        "stats": {"k": 5, "d": 2, "a": 1, "dmg": 3200.0, "heal": 0.0, "mit": 5100.0},
        "hero_panels": [],
        "in_party": false,
        "joined_at": 1774119863346,
        "left_at": null
      }
    }
  }
}
```

This event fires frequently (on every roster update, map change, round start/end, etc.). Companion apps should debounce or diff as needed.

---

### `mcp_submitted`

Emitted after a match is successfully submitted to the MCP server (requires `--mcp` flag).

```json
{
  "type": "mcp_submitted",
  "data": {
    "match_id": "abc-123-def"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `match_id` | `string?` | MCP match ID, if returned by the server |

---

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
| `submit_win` | Manually set result as VICTORY and trigger match end |
| `submit_loss` | Manually set result as DEFEAT and trigger match end |
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
state         → active: true (listening started)
match_state   → match started (Overwolf MatchStartEvent)
match_state   → roster/map/mode updates (each mutation emits full state)
hero_switch   → initial hero assignments (subtitle OCR or Overwolf roster)
tab_capture   → player presses Tab
hero_crop     → hero panel detected in Tab screenshot
match_state   → hero swaps, round starts/ends, stat updates
hero_switch   → mid-match hero swaps
player_change → player leaves/joins mid-match
match_state   → match outcome (VICTORY/DEFEAT)
detection     → VICTORY/DEFEAT recognized (subtitle OCR)
match_complete → match ended, summary printed (Overwolf outcome or subtitle detection)
mcp_submitted → match submitted to MCP server (if --mcp enabled)
```

## Configuration

| Env variable | Default | Description |
|-------------|---------|-------------|
| `WS_PORT` | `42685` | WebSocket server port |
