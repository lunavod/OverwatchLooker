"""Overwolf GEP WebSocket receiver.

Runs a WebSocket server on port 28025 that accepts connections from the
OverwatchListener Overwolf app.  All GEP events are parsed into typed
dataclasses and dispatched via callbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import IO, Any, Callable, Protocol

import websockets
from websockets.asyncio.server import Server, ServerConnection

_logger = logging.getLogger("overwatchlooker")

DEFAULT_PORT = 28025


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GameState(Enum):
    LOADING_SCREEN_START = "loading_screen_start"
    GAME_LOADED = "game_loaded"
    MATCH_IN_PROGRESS = "match_in_progress"
    MATCH_ENDED = "match_ended"


class GameType(Enum):
    UNKNOWN = "UNKNOWN"
    UNRANKED = "UNRANKED"
    CUSTOM_GAME = "CUSTOM_GAME"
    PRACTICE = "PRACTICE"
    ARCADE = "ARCADE"
    TUTORIAL = "TUTORIAL"
    SKIRMISH = "SKIRMISH"
    VS_AI = "VS_AI"
    DEATHMATCH = "DEATHMATCH"
    RANKED = "RANKED"
    HERO_MASTERY = "HERO_MASTERY"


class QueueType(Enum):
    OPEN_QUEUE = "OPEN_QUEUE"
    ROLE_QUEUE = "ROLE_QUEUE"


class MatchOutcome(Enum):
    VICTORY = "victory"
    DEFEAT = "defeat"


class HeroRole(Enum):
    TANK = "TANK"
    DAMAGE = "DAMAGE"
    SUPPORT = "SUPPORT"


# ---------------------------------------------------------------------------
# Map / mode lookup tables
# ---------------------------------------------------------------------------

MAP_CODES: dict[str, str] = {
    "212": "King's Row",
    "388": "Watchpoint: Gibraltar",
    "468": "Numbani",
    "687": "Hollywood",
    "707": "Dorado",
    "1207": "Nepal",
    "1467": "Route 66",
    "1634": "Lijiang Tower",
    "1645": "Ilios",
    "1677": "Eichenwalde",
    "1694": "Oasis",
    "1707": "Hollywood Halloween",
    "1713": "King's Row Winter",
    "1719": "Lijiang Tower Lunar New Year",
    "1878": "Junkertown",
    "1886": "Blizzard World",
    "2018": "Busan",
    "2036": "Eichenwalde Halloween",
    "2087": "Circuit Royal",
    "2161": "Rialto",
    "2360": "Paraíso",
    "2628": "Havana",
    "2651": "Blizzard World Winter",
    "2795": "New Queen Street",
    "2868": "Colosseo",
    "2892": "Midtown",
    "3205": "Shambali Monastery",
    "3314": "Antarctic Peninsula",
    "3390": "Suravasa",
    "3411": "Esperança",
    "3603": "New Junk City",
}

MODE_CODES: dict[str, str] = {
    "0003": "Junkensteins Revenge",
    "0007": "CTF",
    "0008": "Meis Snowball Offensive",
    "0009": "Elimination",
    "0015": "Uprising",
    "0016": "Skirmish",
    "0020": "Assault",
    "0021": "Escort",
    "0022": "Hybrid",
    "0023": "Control",
    "0025": "Tutorial",
    "0026": "Uprising All Heroes",
    "0029": "Team Deathmatch",
    "0030": "Deathmatch",
    "0032": "Lucioball",
    "0037": "Retribution",
    "0041": "Yeti Hunter",
    "0042": "Halloween Holdout Endless",
    "0061": "Calypso Heromode",
    "0064": "Push",
    "0066": "Story Missions",
    "0067": "Storm Rising",
    "0074": "Survivor",
    "0077": "Clash",
    "0089": "Snowball Deathmatch",
    "0090": "Practice Range",
    "0109": "Flashpoint",
    "0112": "Bounty Hunter",
    "0165": "Hero Mastery - Solo",
    "0186": "Hero Mastery CO-OP",
}


# ---------------------------------------------------------------------------
# Typed event dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RosterEntry:
    """A single player in the match roster."""
    player_name: str
    battlenet_tag: str
    is_local: bool
    is_teammate: bool
    hero_name: str  # may be empty for enemies
    hero_role: str  # may be empty for enemies
    team: int
    kills: int
    deaths: int
    assists: int
    damage: float
    healed: float
    mitigated: float

    @classmethod
    def from_json(cls, raw: str) -> RosterEntry:
        d = json.loads(raw)
        return cls(
            player_name=d.get("player_name") or "",
            battlenet_tag=d.get("battlenet_tag") or "",
            is_local=bool(d.get("is_local", False)),
            is_teammate=bool(d.get("is_teammate", False)),
            hero_name=d.get("hero_name") or "",
            hero_role=d.get("hero_role") or "",
            team=_safe_int(d.get("team", 0)),
            kills=_safe_int(d.get("kills", 0)),
            deaths=_safe_int(d.get("deaths", 0)),
            assists=_safe_int(d.get("assists", 0)),
            damage=float(d.get("damage") or 0),
            healed=float(d.get("healed") or 0),
            mitigated=float(d.get("mitigated") or 0),
        )


# -- Info update events --

@dataclass
class GameStateUpdate:
    """game_info.game_state changed."""
    state: GameState
    timestamp: int


@dataclass
class GameModeUpdate:
    """game_info.game_mode changed."""
    code: str
    name: str  # resolved from MODE_CODES
    timestamp: int


@dataclass
class BattleTagUpdate:
    """game_info.battle_tag — local player's tag."""
    battle_tag: str
    timestamp: int


@dataclass
class GameTypeUpdate:
    """match_info.game_type (delivered under game_info feature)."""
    game_type: GameType
    timestamp: int


@dataclass
class QueueTypeUpdate:
    """match_info.game_queue_type."""
    queue_type: QueueType
    timestamp: int


@dataclass
class MapUpdate:
    """match_info.map changed."""
    code: str
    name: str  # resolved from MAP_CODES
    timestamp: int


@dataclass
class PseudoMatchIdUpdate:
    """match_info.pseudo_match_id — Overwolf-generated session UUID."""
    pseudo_match_id: str
    timestamp: int


@dataclass
class MatchOutcomeUpdate:
    """match_info.match_outcome — victory/defeat."""
    outcome: MatchOutcome
    timestamp: int


@dataclass
class EliminationsUpdate:
    """kill.eliminations — cumulative."""
    count: int
    timestamp: int


@dataclass
class ObjectiveKillsUpdate:
    """kill.objective_kills — cumulative."""
    count: int
    timestamp: int


@dataclass
class DeathsUpdate:
    """death.deaths — cumulative."""
    count: int
    timestamp: int


@dataclass
class AssistsUpdate:
    """assist.assist — cumulative."""
    count: int
    timestamp: int


@dataclass
class RosterUpdate:
    """One roster slot updated."""
    slot: int  # 0-9
    entry: RosterEntry
    timestamp: int


# -- Discrete events --

@dataclass
class EliminationEvent:
    """Fired on each kill."""
    total: int  # cumulative at this point
    timestamp: int


@dataclass
class DeathEvent:
    """Fired on each death."""
    total: int
    timestamp: int


@dataclass
class AssistEvent:
    """Fired on each assist."""
    total: int
    timestamp: int


@dataclass
class MatchStartEvent:
    """Match started."""
    timestamp: int


@dataclass
class MatchEndEvent:
    """Match ended."""
    timestamp: int


@dataclass
class RoundStartEvent:
    """New round started."""
    timestamp: int


@dataclass
class RoundEndEvent:
    """Round ended."""
    timestamp: int


@dataclass
class RespawnEvent:
    """Player respawned."""
    timestamp: int


@dataclass
class ReviveEvent:
    """Player was revived."""
    timestamp: int


@dataclass
class GameStatusEvent:
    """Overwatch 2 launched or closed (synthetic from game detection)."""
    running: bool
    timestamp: int


# Union of all typed events
OverwolfEvent = (
    GameStateUpdate | GameModeUpdate | BattleTagUpdate | GameTypeUpdate |
    QueueTypeUpdate | MapUpdate | PseudoMatchIdUpdate | MatchOutcomeUpdate |
    EliminationsUpdate | ObjectiveKillsUpdate | DeathsUpdate | AssistsUpdate |
    RosterUpdate |
    EliminationEvent | DeathEvent | AssistEvent |
    MatchStartEvent | MatchEndEvent | RoundStartEvent | RoundEndEvent |
    RespawnEvent | ReviveEvent |
    GameStatusEvent
)


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------

class OverwolfListener(Protocol):
    """Protocol for receiving typed Overwolf events."""
    def on_overwolf_event(self, event: OverwolfEvent) -> None: ...


# ---------------------------------------------------------------------------
# Message parser
# ---------------------------------------------------------------------------

def _parse_message(raw: dict[str, Any]) -> list[OverwolfEvent]:
    """Parse a raw WebSocket message into typed events."""
    msg_type = raw.get("type")
    ts = raw.get("timestamp", 0)
    data = raw.get("data", {})
    events: list[OverwolfEvent] = []

    if msg_type == "game_status":
        events.append(GameStatusEvent(running=bool(data.get("running")), timestamp=ts))

    elif msg_type == "event":
        for ev in data.get("events", []):
            name = ev.get("name", "")
            ev_data = ev.get("data")
            if name == "elimination":
                events.append(EliminationEvent(total=_safe_int(ev_data), timestamp=ts))
            elif name == "death":
                events.append(DeathEvent(total=_safe_int(ev_data), timestamp=ts))
            elif name == "assist":
                events.append(AssistEvent(total=_safe_int(ev_data), timestamp=ts))
            elif name == "match_start":
                events.append(MatchStartEvent(timestamp=ts))
            elif name == "match_end":
                events.append(MatchEndEvent(timestamp=ts))
            elif name == "round_start":
                events.append(RoundStartEvent(timestamp=ts))
            elif name == "round_end":
                events.append(RoundEndEvent(timestamp=ts))
            elif name == "respawn":
                events.append(RespawnEvent(timestamp=ts))
            elif name == "revive":
                events.append(ReviveEvent(timestamp=ts))
            else:
                _logger.debug(f"Unknown Overwolf event: {name}")

    elif msg_type == "info_update":
        info = data.get("info", {})

        # game_info category
        gi = info.get("game_info", {})
        if "game_state" in gi:
            try:
                events.append(GameStateUpdate(
                    state=GameState(gi["game_state"]), timestamp=ts))
            except ValueError:
                _logger.debug(f"Unknown game_state: {gi['game_state']}")
        if "game_mode" in gi:
            code = gi["game_mode"]
            padded = code.zfill(4) if code else code
            events.append(GameModeUpdate(
                code=code, name=MODE_CODES.get(padded, f"Unknown ({code})"), timestamp=ts))
        if "battle_tag" in gi:
            events.append(BattleTagUpdate(battle_tag=gi["battle_tag"], timestamp=ts))

        # match_info category (some delivered under game_info feature)
        mi = info.get("match_info", {})
        if "game_type" in mi:
            try:
                events.append(GameTypeUpdate(
                    game_type=GameType(mi["game_type"]), timestamp=ts))
            except ValueError:
                _logger.debug(f"Unknown game_type: {mi['game_type']}")
        if "game_queue_type" in mi:
            try:
                events.append(QueueTypeUpdate(
                    queue_type=QueueType(mi["game_queue_type"]), timestamp=ts))
            except ValueError:
                _logger.debug(f"Unknown queue_type: {mi['game_queue_type']}")
        if "map" in mi:
            code = mi["map"]
            events.append(MapUpdate(
                code=code, name=MAP_CODES.get(code, f"Unknown ({code})"), timestamp=ts))
        if "pseudo_match_id" in mi:
            events.append(PseudoMatchIdUpdate(
                pseudo_match_id=mi["pseudo_match_id"], timestamp=ts))
        if "match_outcome" in mi:
            try:
                events.append(MatchOutcomeUpdate(
                    outcome=MatchOutcome(mi["match_outcome"]), timestamp=ts))
            except ValueError:
                _logger.debug(f"Unknown match_outcome: {mi['match_outcome']}")

        # kill category
        ki = info.get("kill", {})
        if "eliminations" in ki:
            events.append(EliminationsUpdate(
                count=_safe_int(ki["eliminations"]), timestamp=ts))
        if "objective_kills" in ki:
            events.append(ObjectiveKillsUpdate(
                count=_safe_int(ki["objective_kills"]), timestamp=ts))

        # death category
        de = info.get("death", {})
        if "deaths" in de:
            events.append(DeathsUpdate(
                count=_safe_int(de["deaths"]), timestamp=ts))

        # assist category
        ai = info.get("assist", {})
        if "assist" in ai:
            events.append(AssistsUpdate(
                count=_safe_int(ai["assist"]), timestamp=ts))

        # roster category
        ro = info.get("roster", {})
        for key, val in ro.items():
            if key.startswith("roster_"):
                try:
                    slot = int(key.split("_", 1)[1])
                    entry = RosterEntry.from_json(val)
                    events.append(RosterUpdate(slot=slot, entry=entry, timestamp=ts))
                except (ValueError, json.JSONDecodeError, KeyError) as e:
                    _logger.debug(f"Failed to parse {key}: {e}")

    else:
        _logger.debug(f"Unknown Overwolf message type: {msg_type}")

    return events


def _safe_int(v: Any) -> int:
    if v is None:
        return 0
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

class OverwolfReceiver:
    """WebSocket server that receives events from OverwatchListener.

    Runs on port 28025 (matching OverwatchListener's hardcoded WS_URL).
    Parses all GEP messages into typed dataclasses and dispatches them
    to registered listeners.
    """

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        self._port = port
        self._host = host
        self._listeners: list[Callable[[OverwolfEvent], None]] = []
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Server | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def add_listener(self, callback: Callable[[OverwolfEvent], None]) -> None:
        """Register a callback for all parsed Overwolf events."""
        self._listeners.append(callback)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except RuntimeError:
            pass

    async def _serve(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
        )
        _logger.info(f"Overwolf receiver listening on ws://{self._host}:{self._port}")

        # Keep the server running
        try:
            await asyncio.Future()  # run forever
        except asyncio.CancelledError:
            pass
        finally:
            if self._server:
                self._server.close()
                await self._server.wait_closed()

    async def _handler(self, ws: ServerConnection) -> None:
        self._connected = True
        _logger.info("OverwatchListener connected")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    _logger.debug(f"Invalid JSON from Overwolf: {raw!r:.200}")
                    continue
                events = _parse_message(msg)
                for event in events:
                    self._dispatch(event)
        finally:
            self._connected = False
            _logger.info("OverwatchListener disconnected")

    def _dispatch(self, event: OverwolfEvent) -> None:
        for cb in self._listeners:
            try:
                cb(event)
            except Exception:
                _logger.exception(f"Error in Overwolf listener callback for {type(event).__name__}")

    def stop(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)


# ---------------------------------------------------------------------------
# Event queue (thread-safe, drains between ticks)
# ---------------------------------------------------------------------------

class OverwolfEventQueue:
    """Thread-safe queue: Overwolf receiver pushes, tick system drains."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: list[OverwolfEvent] = []

    def push(self, event: OverwolfEvent) -> None:
        with self._lock:
            self._buf.append(event)

    def drain(self) -> list[OverwolfEvent]:
        with self._lock:
            events = self._buf
            self._buf = []
            return events


# ---------------------------------------------------------------------------
# Serialization (for recording JSONL)
# ---------------------------------------------------------------------------

# Registry: class name -> class
_EVENT_CLASSES: dict[str, type] = {
    cls.__name__: cls
    for cls in [
        GameStateUpdate, GameModeUpdate, BattleTagUpdate, GameTypeUpdate,
        QueueTypeUpdate, MapUpdate, PseudoMatchIdUpdate, MatchOutcomeUpdate,
        EliminationsUpdate, ObjectiveKillsUpdate, DeathsUpdate, AssistsUpdate,
        RosterUpdate,
        EliminationEvent, DeathEvent, AssistEvent,
        MatchStartEvent, MatchEndEvent, RoundStartEvent, RoundEndEvent,
        RespawnEvent, ReviveEvent,
        GameStatusEvent,
    ]
}

# Enum fields that need special handling
_ENUM_FIELDS: dict[str, dict[str, type]] = {
    "GameStateUpdate": {"state": GameState},
    "GameTypeUpdate": {"game_type": GameType},
    "QueueTypeUpdate": {"queue_type": QueueType},
    "MatchOutcomeUpdate": {"outcome": MatchOutcome},
}


def serialize_event(event: OverwolfEvent, frame: int) -> str:
    """Serialize an event to a JSONL line: {"frame": N, "cls": "...", ...fields}."""
    cls_name = type(event).__name__
    d: dict[str, Any] = {"frame": frame, "cls": cls_name}

    if isinstance(event, RosterUpdate):
        d["slot"] = event.slot
        d["entry"] = asdict(event.entry)
        d["timestamp"] = event.timestamp
    else:
        for f in fields(event):
            val = getattr(event, f.name)
            if isinstance(val, Enum):
                val = val.value
            d[f.name] = val

    return json.dumps(d, separators=(",", ":"))


def deserialize_event(line: str) -> tuple[int, OverwolfEvent]:
    """Deserialize a JSONL line back to (frame, event)."""
    d = json.loads(line)
    frame = d.pop("frame")
    cls_name = d.pop("cls")
    cls = _EVENT_CLASSES.get(cls_name)
    if cls is None:
        raise ValueError(f"Unknown event class: {cls_name}")

    # Reconstruct RosterEntry
    if cls_name == "RosterUpdate":
        d["entry"] = RosterEntry(**d["entry"])
        return frame, cls(**d)

    # Reconstruct enum values
    enum_map = _ENUM_FIELDS.get(cls_name, {})
    for field_name, enum_cls in enum_map.items():
        if field_name in d:
            d[field_name] = enum_cls(d[field_name])

    return frame, cls(**d)


# ---------------------------------------------------------------------------
# JSONL recording writer
# ---------------------------------------------------------------------------

class OverwolfRecordingWriter:
    """Writes Overwolf events to a JSONL file during recording."""

    def __init__(self, path: Path) -> None:
        self._file: IO[str] = open(path, "w", encoding="utf-8")
        _logger.info(f"Overwolf recording: {path}")

    def write(self, event: OverwolfEvent, frame: int) -> None:
        line = serialize_event(event, frame)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def load_overwolf_events(path: Path) -> list[tuple[int, OverwolfEvent]]:
    """Load events from a recording JSONL file. Returns list of (frame, event)."""
    events: list[tuple[int, OverwolfEvent]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(deserialize_event(line))
            except Exception as e:
                _logger.debug(f"Skipping bad overwolf event line: {e}")
    events.sort(key=lambda x: x[0])
    return events
