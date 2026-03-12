# MCP Schema Update: Per-Hero Stats with Hero Switches

## What changed

The client-side `submit_match` payload has changed how hero data is sent per player. The old `hero` field (single object or null) has been **replaced** by a `heroes` array on each player. The top-level `extra_hero_stats` field is also **gone** — everything is merged into the per-player `heroes` arrays before submission.

## Schema change for the player object

**Remove:**
- `hero` (object | null) — no longer sent

**Add to the player object:**
```json
"heroes": {
  "type": "array",
  "description": "All heroes this player played during the match, with optional stats and switch timestamps.",
  "items": {
    "type": "object",
    "properties": {
      "hero_name": {
        "type": "string",
        "description": "Hero name (e.g. 'Reinhardt', 'Moira', 'D.Va')"
      },
      "started_at": {
        "type": "array",
        "items": { "type": "integer" },
        "description": "Seconds from match start when this hero was played. Multiple entries mean the player switched away and back to this hero."
      },
      "stats": {
        "type": "array",
        "description": "Hero-specific stat entries from the Tab screen's right panel. Empty array if stats are not available for this hero (non-self players, or self-player heroes whose stat panel wasn't captured).",
        "items": {
          "type": "object",
          "properties": {
            "label": { "type": "string", "description": "Stat label, e.g. 'Weapon Accuracy', 'Charge Kills'" },
            "value": { "type": "string", "description": "Stat value as displayed, e.g. '49%', '3', '00:37'" },
            "is_featured": { "type": "boolean", "description": "True for the big highlighted stat at the top of the hero panel" }
          },
          "required": ["label", "value", "is_featured"]
        }
      }
    },
    "required": ["hero_name", "started_at", "stats"]
  }
}
```

## What each field means

| Field | Description |
|-------|-------------|
| `heroes` | Array of all heroes this player played. Always present, even if only one hero (no switches). |
| `hero_name` | The hero's canonical name. |
| `started_at` | When the player started playing this hero, as seconds from match start. Comes from subtitle OCR tracking of in-game hero switch announcements. `[0]` = played from the start. `[0, 400]` = played at start, switched away, came back at 4:00. Empty `[]` only if switch timing is unknown (e.g. subtitle tracking wasn't active). |
| `stats` | Hero-specific stats from the Tab screen's right-side panel. Only populated for the `is_self=true` player (the panel is only visible for your own hero). Empty `[]` for all other players, and for self-player heroes whose panel wasn't captured. |
| `is_featured` | The one big highlighted stat at the top of the hero panel (e.g. "17 PLAYERS SAVED"). Each hero has at most one. |

## Data availability by player type

| Player | `heroes` populated? | `started_at` populated? | `stats` populated? |
|--------|---------------------|------------------------|--------------------|
| Self (is_self=true) | Yes, all heroes played | Yes, from subtitle tracking | Yes, for heroes whose Tab panel was captured |
| Ally (non-self) | Yes, from subtitle tracking | Yes | No (always `[]`) |
| Enemy | Yes, from subtitle tracking | Yes | No (always `[]`) |

## Example payload (3 players)

Self-player who switched Reinhardt → Juno → Moira, with stats for all three:
```json
{
  "player_name": "LUNAVOD",
  "is_self": true,
  "heroes": [
    {"hero_name": "Reinhardt", "started_at": [0], "stats": [{"label": "Charge Kills", "value": "3", "is_featured": false}]},
    {"hero_name": "Juno", "started_at": [120], "stats": [{"label": "Weapon Accuracy", "value": "0%", "is_featured": false}]},
    {"hero_name": "Moira", "started_at": [280], "stats": [{"label": "Players Saved", "value": "17", "is_featured": true}]}
  ]
}
```

Ally who switched D.Va → Orisa (no stats — panel not visible for other players):
```json
{
  "player_name": "ENRICOPUCCI",
  "is_self": false,
  "heroes": [
    {"hero_name": "D.Va", "started_at": [0], "stats": []},
    {"hero_name": "Orisa", "started_at": [200], "stats": []}
  ]
}
```

Enemy with one hero:
```json
{
  "player_name": "M4K",
  "is_self": false,
  "heroes": [
    {"hero_name": "Tracer", "started_at": [0], "stats": []}
  ]
}
```

## Backward compatibility

- The `heroes` array may be **absent** on older submissions (before this change). Treat missing `heroes` the same as the old `hero` field — check for whichever is present.
- The old `hero` field is no longer sent by the client. If you were storing it, you can migrate: `hero` becomes a single-element `heroes` array with `started_at: []` and the same stats.
