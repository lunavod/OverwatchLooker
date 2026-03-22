"""Centralized match state: single source of truth during a match.

All systems (Overwolf, Subtitle OCR, Chat OCR, Tab capture) write to a
shared MatchState instance. At match end the state is snapshotted, printed,
and reset for the next match.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum

from overwatchlooker.overwolf import GameType, QueueType

_logger = logging.getLogger("overwatchlooker")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TeamSide(Enum):
    ALLY = "ALLY"
    ENEMY = "ENEMY"


class MatchResult(Enum):
    VICTORY = "VICTORY"
    DEFEAT = "DEFEAT"


class ResultSource(Enum):
    OVERWOLF = "overwolf:outcome"
    SUBTITLE = "subtitle:ocr"


class HeroSource(Enum):
    OVERWOLF_ROSTER = "overwolf:roster"
    SUBTITLE_OCR = "subtitle:ocr"
    TAB_OCR = "tab:ocr"


class PlayerRole(Enum):
    TANK = "TANK"
    DAMAGE = "DAMAGE"
    SUPPORT = "SUPPORT"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StatsSnapshot:
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    damage: float = 0.0
    healing: float = 0.0
    mitigation: float = 0.0


@dataclass
class HeroSwap:
    hero: str  # Title Case hero name
    detected_at: int  # epoch ms
    source: HeroSource
    stats_at_detection: StatsSnapshot | None = None


@dataclass
class HeroPanel:
    hero_name: str
    crop_png: bytes
    ocr_stats: list[dict] = field(default_factory=list)


@dataclass
class PlayerState:
    player_name: str  # UPPERCASE display name
    battletag: str = ""  # full Tag#1234 from Overwolf
    team: int | None = None  # raw team id (0 or 1) from Overwolf
    team_side: TeamSide | None = None  # resolved ALLY/ENEMY
    is_local: bool = False
    role: PlayerRole | None = None  # from Overwolf (allies only)
    slot: int | None = None  # roster_0..roster_9
    hero_swaps: list[HeroSwap] = field(default_factory=list)
    stats: StatsSnapshot | None = None  # latest cumulative from Overwolf
    hero_panels: list[HeroPanel] = field(default_factory=list)
    in_party: bool = False  # green party indicator on scoreboard
    joined_at: int | None = None  # epoch ms
    left_at: int | None = None  # epoch ms

    @property
    def current_hero(self) -> str | None:
        """Last hero from hero_swaps, or None."""
        return self.hero_swaps[-1].hero if self.hero_swaps else None


@dataclass
class RoundInfo:
    started_at: int  # epoch ms
    ended_at: int | None = None


@dataclass
class TabScreenshot:
    png_bytes: bytes
    sim_time: float
    filename: str


@dataclass
class HeroTabCapture:
    """A tab screenshot captured while playing a specific hero."""
    hero_name: str
    png_bytes: bytes
    tick: int
    filename: str


@dataclass
class MatchState:
    # Match-level (from Overwolf)
    map_name: str = ""
    map_code: str = ""
    mode: str = ""
    mode_code: str = ""
    game_type: GameType | None = None
    queue_type: QueueType | None = None
    result: MatchResult | None = None
    result_source: ResultSource | None = None
    started_at: int | None = None  # epoch ms from MatchStartEvent
    ended_at: int | None = None  # epoch ms from MatchEndEvent
    rounds: list[RoundInfo] = field(default_factory=list)
    pseudo_match_id: str = ""

    # Rank (from tab screenshot OCR)
    rank_min: str = ""       # e.g. "Bronze 2"
    rank_max: str = ""       # e.g. "Gold 1"
    is_wide_match: bool | None = None

    # Backfill detection
    is_backfill: bool = False

    # Hero bans (from tab screenshot template matching)
    hero_bans: list[str] = field(default_factory=list)

    # From Tab capture — one per hero (latest with visible panel), plus latest raw for rank
    hero_tabs: dict[str, HeroTabCapture] = field(default_factory=dict)  # hero_name -> capture
    latest_tab: TabScreenshot | None = None  # most recent tab screenshot (for rank detection)

    # Players keyed by UPPERCASE player_name
    players: dict[str, PlayerState] = field(default_factory=dict)

    # Internal tracking
    _local_team: int | None = None  # raw team id of local player
    _analysis_triggered: bool = False  # prevent double trigger

    def get_or_create_player(self, name: str) -> PlayerState:
        """Get existing player or create a new one. Key is UPPERCASE name."""
        key = name.upper()
        if key not in self.players:
            self.players[key] = PlayerState(player_name=key)
        return self.players[key]

    @property
    def local_player(self) -> PlayerState | None:
        """Return the player with is_local=True, or None."""
        for p in self.players.values():
            if p.is_local:
                return p
        return None

    @property
    def duration(self) -> int | None:
        """Match duration in ms, measured from first round start to match end.

        Uses first RoundStartEvent as the gameplay start (excludes hero select),
        falling back to MatchStartEvent if no rounds recorded.
        """
        if self.ended_at is None:
            return None
        start = self.rounds[0].started_at if self.rounds else self.started_at
        if start is None:
            return None
        return self.ended_at - start

    def resolve_team_sides(self) -> None:
        """Derive TeamSide for all players based on _local_team."""
        if self._local_team is None:
            return
        for p in self.players.values():
            if p.team is not None:
                p.team_side = (
                    TeamSide.ALLY if p.team == self._local_team
                    else TeamSide.ENEMY
                )

    def snapshot(self) -> MatchState:
        """Return a deep copy for analysis, mark analysis as triggered."""
        self._analysis_triggered = True
        return deepcopy(self)

    def dump_to_log(self, label: str = "MatchState") -> None:
        """Log the full match state as JSON (skipping image bytes)."""
        try:
            _logger.info(f"[{label}] {json.dumps(self._to_dict(), separators=(',', ':'))}")
        except Exception as e:
            _logger.warning(f"[{label}] dump failed: {e}")

    def _to_dict(self) -> dict:
        """Serialize to a JSON-safe dict, omitting binary data."""
        def _enum_val(v):
            return v.value if isinstance(v, Enum) else v

        players = {}
        for name, p in self.players.items():
            players[name] = {
                "battletag": p.battletag,
                "team": p.team,
                "team_side": _enum_val(p.team_side),
                "is_local": p.is_local,
                "role": _enum_val(p.role),
                "slot": p.slot,
                "hero_swaps": [
                    {"hero": s.hero, "at": s.detected_at,
                     "source": _enum_val(s.source),
                     "stats": ({"k": s.stats_at_detection.kills,
                                "d": s.stats_at_detection.deaths,
                                "a": s.stats_at_detection.assists,
                                "dmg": s.stats_at_detection.damage,
                                "heal": s.stats_at_detection.healing,
                                "mit": s.stats_at_detection.mitigation}
                               if s.stats_at_detection else None)}
                    for s in p.hero_swaps
                ],
                "stats": ({"k": p.stats.kills, "d": p.stats.deaths,
                           "a": p.stats.assists, "dmg": p.stats.damage,
                           "heal": p.stats.healing, "mit": p.stats.mitigation}
                          if p.stats else None),
                "hero_panels": [hp.hero_name for hp in p.hero_panels],
                "in_party": p.in_party,
                "joined_at": p.joined_at,
                "left_at": p.left_at,
            }

        return {
            "map": self.map_name, "map_code": self.map_code,
            "mode": self.mode, "mode_code": self.mode_code,
            "game_type": _enum_val(self.game_type),
            "queue_type": _enum_val(self.queue_type),
            "result": _enum_val(self.result),
            "result_source": _enum_val(self.result_source),
            "started_at": self.started_at, "ended_at": self.ended_at,
            "duration_ms": self.duration,
            "rounds": [{"started": r.started_at, "ended": r.ended_at}
                       for r in self.rounds],
            "pseudo_match_id": self.pseudo_match_id,
            "rank_min": self.rank_min or None,
            "rank_max": self.rank_max or None,
            "is_wide_match": self.is_wide_match,
            "is_backfill": self.is_backfill,
            "hero_bans": self.hero_bans or None,
            "local_team": self._local_team,
            "hero_tabs": {name: {"hero": c.hero_name, "tick": c.tick, "file": c.filename}
                         for name, c in self.hero_tabs.items()},
            "latest_tab": self.latest_tab.filename if self.latest_tab else None,
            "players": players,
        }


# ---------------------------------------------------------------------------
# Console formatter
# ---------------------------------------------------------------------------

def _format_duration(ms: int | None) -> str:
    if ms is None:
        return "??:??"
    total_s = ms // 1000
    return f"{total_s // 60}:{total_s % 60:02d}"


_ROLE_ORDER = {PlayerRole.TANK: 0, PlayerRole.DAMAGE: 1, PlayerRole.SUPPORT: 2}


def _role_label(role: PlayerRole | None) -> str:
    if role is None:
        return "     "
    return f"{role.value:<5s}"


def _player_line(p: PlayerState) -> str:
    """Format one player line for console output."""
    role = _role_label(p.role)
    name = p.battletag if p.battletag else p.player_name
    name = f"{name:<20s}"

    # Hero history
    if p.hero_swaps:
        heroes_parts: list[str] = []
        for i, swap in enumerate(p.hero_swaps):
            if i == 0:
                heroes_parts.append(swap.hero)
            else:
                t = _format_duration(swap.detected_at - (p.hero_swaps[0].detected_at))
                heroes_parts.append(f"{swap.hero} ({t})")
        hero_str = " -> ".join(heroes_parts)
    else:
        hero_str = "(hero unknown)"
    hero_str = f"{hero_str:<35s}"

    # Stats
    if p.stats:
        s = p.stats
        stats_str = (f"{s.kills}/{s.deaths}/{s.assists}   "
                     f"{s.damage:.0f} dmg  {s.healing:.0f} heal  {s.mitigation:.0f} mit")
    else:
        stats_str = ""

    party = " [P]" if p.in_party else ""
    return f"  {role} {name} {hero_str} {stats_str}{party}"


def format_match_state(match: MatchState) -> str:
    """Format a completed MatchState as a human-readable console string."""
    lines: list[str] = []

    # Header
    lines.append("")
    lines.append("\u2550\u2550\u2550 MATCH COMPLETE \u2550\u2550\u2550")

    # Map / mode
    map_str = match.map_name or "Unknown Map"
    mode_str = match.mode or ""
    game_type_str = ""
    if match.game_type:
        gt = match.game_type.value.replace("_", " ").title()
        if gt == "Ranked":
            gt = "Competitive"
        game_type_str = gt
    queue_str = ""
    if match.queue_type:
        queue_str = match.queue_type.value.replace("_", " ").title()

    info_parts = [p for p in [map_str, mode_str, game_type_str, queue_str] if p]
    lines.append(f"Map: {' | '.join(info_parts)}")

    # Result + duration
    result_str = match.result.value if match.result else "UNKNOWN"
    dur_str = _format_duration(match.duration)
    backfill_str = " (BACKFILL)" if match.is_backfill else ""
    lines.append(f"Result: {result_str} | Duration: {dur_str}{backfill_str}")

    # Rank
    if match.rank_min and match.rank_max:
        rank_str = f"Rank: {match.rank_min} — {match.rank_max}"
        if match.is_wide_match:
            rank_str += " (WIDE)"
        lines.append(rank_str)

    # Hero bans
    if match.hero_bans:
        lines.append(f"Bans: {', '.join(match.hero_bans)}")

    # Rounds
    if match.rounds:
        round_durs = []
        for r in match.rounds:
            if r.ended_at is not None:
                round_durs.append(_format_duration(r.ended_at - r.started_at))
            else:
                round_durs.append("??:??")
        lines.append(f"Rounds: {len(match.rounds)} ({', '.join(round_durs)})")

    # Group players by team side
    allies: list[PlayerState] = []
    enemies: list[PlayerState] = []
    unknown_team: list[PlayerState] = []
    for p in match.players.values():
        if p.team_side == TeamSide.ALLY:
            allies.append(p)
        elif p.team_side == TeamSide.ENEMY:
            enemies.append(p)
        else:
            unknown_team.append(p)

    def _sort_key(p: PlayerState) -> tuple[int, str]:
        return (_ROLE_ORDER.get(p.role, 99) if p.role else 99, p.player_name)

    if allies:
        team_label = f"ALLY (Team {match._local_team})" if match._local_team is not None else "ALLY"
        lines.append("")
        lines.append(f"\u2500\u2500 {team_label} \u2500\u2500")
        for p in sorted(allies, key=_sort_key):
            lines.append(_player_line(p))

    if enemies:
        enemy_team = None
        for p in enemies:
            if p.team is not None:
                enemy_team = p.team
                break
        team_label = f"ENEMY (Team {enemy_team})" if enemy_team is not None else "ENEMY"
        lines.append("")
        lines.append(f"\u2500\u2500 {team_label} \u2500\u2500")
        for p in sorted(enemies, key=_sort_key):
            lines.append(_player_line(p))

    if unknown_team:
        lines.append("")
        lines.append("\u2500\u2500 UNKNOWN TEAM \u2500\u2500")
        for p in sorted(unknown_team, key=_sort_key):
            lines.append(_player_line(p))

    # Hero stats (from panel OCR)
    local = match.local_player
    if local and local.hero_panels:
        for hp in local.hero_panels:
            if hp.ocr_stats:
                lines.append("")
                lines.append(f"\u2500\u2500 {hp.hero_name} Stats \u2500\u2500")
                for stat in hp.ocr_stats:
                    prefix = "\u2605 " if stat.get("is_featured") else "  "
                    lines.append(f"{prefix}{stat.get('label', '?')}: {stat.get('value', '?')}")

    lines.append("\u2550" * 23)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP payload builder
# ---------------------------------------------------------------------------

_MODE_TO_MCP: dict[str, str] = {
    "Escort": "ESCORT",
    "Hybrid": "HYBRID",
    "Control": "CONTROL",
    "Push": "PUSH",
    "Clash": "CLASH",
    "Flashpoint": "FLASHPOINT",
}


def build_mcp_payload(match: MatchState) -> dict:
    """Convert a MatchState snapshot into the dict expected by mcp_client.submit_match."""
    # Duration as MM:SS
    duration_str = _format_duration(match.duration)

    # Mode: map human-readable to MCP uppercase, fallback to uppercase of whatever we have
    mode = _MODE_TO_MCP.get(match.mode, match.mode.upper() if match.mode else "UNKNOWN")

    # Queue type: RANKED → COMPETITIVE, everything else → QUICKPLAY
    if match.game_type == GameType.RANKED:
        queue_type = "COMPETITIVE"
    else:
        queue_type = "QUICKPLAY"

    # Result
    result = match.result.value if match.result else "UNKNOWN"

    # Build players array
    players: list[dict] = []
    # Determine match start time for relative timestamps
    match_start_ms = (match.rounds[0].started_at if match.rounds
                      else match.started_at or 0)

    # Detect disconnected players: when two players share a slot,
    # the earlier one (by first hero_swap timestamp) disconnected.
    disconnected: set[str] = set()
    slot_owners: dict[int, list[PlayerState]] = {}
    for p in match.players.values():
        if p.slot is not None:
            slot_owners.setdefault(p.slot, []).append(p)
    for slot, owners in slot_owners.items():
        if len(owners) > 1:
            # Keep the one with the latest first hero swap (= the replacement)
            owners.sort(key=lambda o: (
                o.hero_swaps[0].detected_at if o.hero_swaps else 0))
            for old in owners[:-1]:
                disconnected.add(old.player_name)
                _logger.info(f"MCP payload: skipping disconnected player "
                             f"{old.player_name} (slot {slot} taken by "
                             f"{owners[-1].player_name})")

    for p in match.players.values():
        if p.player_name in disconnected:
            continue

        team = p.team_side.value if p.team_side else "ALLY"
        role = p.role.value if p.role else None
        # MCP expects DPS, not DAMAGE
        if role == "DAMAGE":
            role = "DPS"

        player_dict: dict = {
            "team": team,
            "player_name": p.battletag if p.battletag else p.player_name,
            "is_self": p.is_local,
            "in_party": p.in_party,
        }
        if role:
            player_dict["role"] = role

        # Joined at (seconds from match start)
        if p.joined_at is not None:
            player_dict["joined_at"] = max(0, (p.joined_at - match_start_ms) // 1000)

        # Current hero (last swap)
        if p.current_hero:
            player_dict["hero_name"] = p.current_hero

        # Stats
        if p.stats:
            player_dict["eliminations"] = p.stats.kills
            player_dict["deaths"] = p.stats.deaths
            player_dict["assists"] = p.stats.assists
            player_dict["damage"] = int(p.stats.damage)
            player_dict["healing"] = int(p.stats.healing)
            player_dict["mitigation"] = int(p.stats.mitigation)

        # Heroes array (swap history + per-hero panel stats)
        if p.hero_swaps:
            heroes: list[dict] = []
            panel_by_hero: dict[str, HeroPanel] = {}
            for hp in p.hero_panels:
                panel_by_hero[hp.hero_name] = hp

            for swap in p.hero_swaps:
                started_at_sec = max(
                    0, (swap.detected_at - match_start_ms) // 1000)
                hero_entry: dict = {
                    "hero_name": swap.hero,
                    "started_at": [started_at_sec],
                }
                # Attach panel OCR stats if available
                panel = panel_by_hero.get(swap.hero)
                if panel and panel.ocr_stats:
                    hero_entry["stats"] = panel.ocr_stats
                heroes.append(hero_entry)

            player_dict["heroes"] = heroes

            # Swap snapshots (cumulative stats at each hero swap point)
            snapshots: list[dict] = []
            for swap in p.hero_swaps:
                if swap.stats_at_detection:
                    s = swap.stats_at_detection
                    snapshots.append({
                        "time": max(0, (swap.detected_at - match_start_ms) // 1000),
                        "eliminations": s.kills,
                        "assists": s.assists,
                        "deaths": s.deaths,
                        "damage": int(s.damage),
                        "healing": int(s.healing),
                        "mitigation": int(s.mitigation),
                    })
            if snapshots:
                player_dict["swap_snapshots"] = snapshots

        players.append(player_dict)

    data: dict = {
        "map_name": match.map_name or "Unknown",
        "duration": duration_str,
        "mode": mode,
        "queue_type": queue_type,
        "result": result,
        "players": players,
        "is_backfill": match.is_backfill,
    }

    # Rank
    if match.rank_min:
        data["rank_min"] = match.rank_min
    if match.rank_max:
        data["rank_max"] = match.rank_max
    if match.is_wide_match:
        data["is_wide_match"] = True

    # Hero bans
    if match.hero_bans:
        data["banned_heroes"] = match.hero_bans

    return data
