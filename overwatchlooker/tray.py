from __future__ import annotations

import logging
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pystray  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFont

from overwatchlooker.display import print_error, print_status
from overwatchlooker.heroes import edit_distance as _edit_distance
from overwatchlooker.match_state import (
    HeroPanel,
    HeroSource,
    HeroSwap,
    MatchResult,
    MatchState,
    PlayerRole,
    ResultSource,
    RoundInfo,
    StatsSnapshot,
    TabScreenshot,
    build_mcp_payload,
    format_match_state,
)
from overwatchlooker.notification import show_notification
from overwatchlooker.overwolf import (
    DeathEvent,
    GameModeUpdate,
    GameType,
    GameTypeUpdate,
    MapUpdate,
    MatchEndEvent,
    MatchOutcomeUpdate,
    MatchStartEvent,
    PseudoMatchIdUpdate,
    QueueTypeUpdate,
    RosterUpdate,
    RoundEndEvent,
    RoundStartEvent,
)

if TYPE_CHECKING:
    from memoir_capture import CaptureEngine
    from overwatchlooker.overwolf import OverwolfEventQueue, OverwolfEvent, OverwolfReceiver
    from overwatchlooker.tick import ChatSystem, ControlScoreSystem, OverwolfSystem, SubtitleSystem, TeamSideSystem, TickLoop
    from overwatchlooker.ws_server import EventBus

_logger = logging.getLogger("overwatchlooker")

_RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"

# Eagerly load pywintypes on the main thread — pywin32's DLL can fail to load
# in worker threads if it hasn't been initialized yet (transitive dep of mcp SDK).
try:
    import pywintypes  # noqa: F401
except ImportError:
    pass



def _create_icon_image() -> Image.Image:
    """Create a simple 64x64 icon: blue circle with 'OW' text."""
    img = Image.new("RGB", (64, 64), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=(51, 153, 255))
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((14, 18), "OW", fill="white", font=font)
    return img


_TAB_DEBOUNCE = 1.5  # ignore Tab presses within this window of each other
_POST_SUBMIT_COOLDOWN = 30.0  # seconds to ignore tab/crop events after detection
_OW_POLL_INTERVAL = 2.0  # seconds between OW window checks
_OW_STABILITY_TIME = 5.0  # seconds OW must be present before starting engine


_AUTO_RECORDINGS_DIR = Path(__file__).parent.parent / "recordings_auto"


class App:
    def __init__(self, use_transcript: bool = False, replay_source=None,
                 event_bus: EventBus | None = None,
                 overwolf_receiver: OverwolfReceiver | None = None,
                 use_mcp: bool = False,
                 auto_recording: bool = False,
                 auto_recording_tail: int = 60):
        self._active = False
        self._detector: SubtitleSystem | None = None
        self._lock = threading.Lock()
        self._match_state: MatchState = MatchState()
        self._last_match_end_ts: float = 0.0  # monotonic time of last match end
        self._use_transcript = use_transcript
        self._replay_source = replay_source  # ReplaySource for replay mode
        self._engine: CaptureEngine | None = None
        self._tick_loop: TickLoop | None = None
        self._subtitle_system: SubtitleSystem | None = None
        self._chat_system: ChatSystem | None = None
        self._team_side_system: TeamSideSystem | None = None
        self._control_score_system: ControlScoreSystem | None = None
        self._bus = event_bus
        self._overwolf = overwolf_receiver
        self._overwolf_queue: OverwolfEventQueue | None = None
        self._overwolf_system: OverwolfSystem | None = None
        self._icon: pystray.Icon | None = None
        self._cooldown_until_tick: int = 0  # ignore tab/crop events until this tick
        self._poll_thread: threading.Thread | None = None
        self._poll_stop = threading.Event()
        self._recording = False
        self._use_mcp = use_mcp
        self._auto_recording = auto_recording
        self._auto_recording_tail = auto_recording_tail
        self._auto_recording_active = False  # True while auto-recording is in progress
        self._auto_stop_timer: threading.Timer | None = None
        self._pending_analysis = threading.Event()
        self._pending_analysis.set()  # no analysis pending initially
        if event_bus:
            self._register_commands(event_bus)
        if overwolf_receiver:
            from overwatchlooker.overwolf import OverwolfEventQueue
            self._overwolf_queue = OverwolfEventQueue()
            overwolf_receiver.add_listener(self._overwolf_queue.push)
        # Preload OCR models in background so first match end is fast
        threading.Thread(target=self._preload_models, daemon=True).start()

    @staticmethod
    def _preload_models() -> None:
        import io
        import os
        import sys
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["GLOG_minloglevel"] = "2"
        # Suppress `where ccache` noise on Windows by patching subprocess
        import subprocess
        _real_check_output = subprocess.check_output
        def _quiet_check_output(cmd, *args, **kwargs):
            kwargs.setdefault("stderr", subprocess.DEVNULL)
            return _real_check_output(cmd, *args, **kwargs)
        subprocess.check_output = _quiet_check_output
        # Redirect stderr to suppress C-level noise from paddle init
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            from overwatchlooker.hero_panel import preload_models
            preload_models()
        except Exception as e:
            _logger.warning(f"OCR model preload failed: {e}")
        finally:
            sys.stderr = old_stderr
            subprocess.check_output = _real_check_output

    def _register_commands(self, bus: EventBus) -> None:
        """Register command handlers on the event bus."""
        bus.register("start_listening", self._start_listening)
        bus.register("stop_listening", self._stop_listening)
        bus.register("toggle_recording", lambda: self._on_toggle_recording(None, None))  # type: ignore[arg-type]
        bus.register("submit_win", lambda: self._on_submit_tab("VICTORY"))
        bus.register("submit_loss", lambda: self._on_submit_tab("DEFEAT"))
        bus.register("quit", self._ws_quit)

    def _ws_quit(self) -> None:
        """Handle quit command from companion app."""
        _logger.info("Quit requested via WebSocket command")
        self._shutdown()

    def _in_post_detection_cooldown(self) -> bool:
        """Check if tab/crop events should be ignored (post-detection cooldown)."""
        if self._cooldown_until_tick <= 0:
            return False
        if self._tick_loop is None:
            return False
        return self._tick_loop._current_tick < self._cooldown_until_tick

    def _ws_emit(self, event: dict) -> None:
        """Emit an event to the WebSocket bus if enabled."""
        if self._bus:
            self._bus.emit(event)

    def _ws_emit_match_state(self) -> None:
        """Emit the current match state to the WebSocket bus."""
        self._ws_emit({"type": "match_state", "data": self._match_state._to_dict()})

    def _on_overwolf_event(self, event: OverwolfEvent) -> None:
        """Handle a typed event from the Overwolf receiver — update MatchState."""
        _logger.info(f"Overwolf: {type(event).__name__}: {event}")
        ms = self._match_state

        if isinstance(event, MatchStartEvent):
            carried = [a for a in ("map_name", "mode", "game_type", "queue_type")
                       if getattr(ms, a)]
            _logger.info(f"MatchStart: creating new MatchState "
                         f"(carrying over: {carried}, "
                         f"buffered players: {len(ms.players)})")
            self._match_state = MatchState(started_at=event.timestamp)
            # Carry over any pre-match info that arrived before MatchStartEvent
            new = self._match_state
            for attr in ("map_name", "map_code", "mode", "mode_code",
                         "game_type", "queue_type", "pseudo_match_id"):
                old_val = getattr(ms, attr)
                new_val = getattr(new, attr)
                if old_val and not new_val:
                    setattr(new, attr, old_val)
            # Carry over players (roster updates may arrive before match_start)
            if ms.players and not new.players:
                new.players = ms.players
                new._local_team = ms._local_team
            ms = new
            new.dump_to_log("MatchStart")
            if self._team_side_system is not None:
                self._team_side_system.start()
            self._auto_recording_start()
            self._ws_emit_match_state()
            return

        if isinstance(event, MatchEndEvent):
            _logger.info(f"MatchEnd: ts={event.timestamp}")
            ms.ended_at = event.timestamp
            self._ws_emit_match_state()
            self._trigger_match_end()
            return

        if isinstance(event, MatchOutcomeUpdate):
            from overwatchlooker.overwolf import MatchOutcome
            _logger.info(f"MatchOutcome: {event.outcome.value}")
            if event.outcome == MatchOutcome.VICTORY:
                ms.result = MatchResult.VICTORY
            else:
                ms.result = MatchResult.DEFEAT
            ms.result_source = ResultSource.OVERWOLF
            if ms.ended_at is None:
                ms.ended_at = event.timestamp
            self._ws_emit_match_state()
            # If no MatchEndEvent arrives, outcome alone can trigger analysis
            if not ms._analysis_triggered:
                self._trigger_match_end()
            return

        if isinstance(event, MapUpdate):
            ms.map_name = event.name
            ms.map_code = event.code
            # Notify about unknown map codes so we can add them
            if event.name.startswith("Unknown ("):
                _REAL_MODES = {"Escort", "Hybrid", "Control", "Push", "Clash", "Flashpoint"}
                if ms.mode in _REAL_MODES:
                    show_notification("OverwatchLooker",
                                      f"Unknown map code {event.code} (mode={ms.mode}). "
                                      f"Please report!")
        elif isinstance(event, GameModeUpdate):
            ms.mode = event.name
            ms.mode_code = event.code
            if (event.name == "Control" and self._control_score_system
                    and ms.game_type != GameType.RANKED):
                self._control_score_system.start()
            if ms.map_name.startswith("Unknown ("):
                _REAL_MODES = {"Escort", "Hybrid", "Control", "Push", "Clash", "Flashpoint"}
                if event.name in _REAL_MODES:
                    show_notification("OverwatchLooker",
                                      f"Unknown map code {ms.map_code} (mode={event.name}). "
                                      f"Please report!")
        elif isinstance(event, GameTypeUpdate):
            ms.game_type = event.game_type
            # Start control score if mode already known and not ranked
            if (ms.mode == "Control" and self._control_score_system
                    and event.game_type != GameType.RANKED
                    and not self._control_score_system._active):
                self._control_score_system.start()
        elif isinstance(event, QueueTypeUpdate):
            ms.queue_type = event.queue_type
        elif isinstance(event, PseudoMatchIdUpdate):
            ms.pseudo_match_id = event.pseudo_match_id
        elif isinstance(event, RoundStartEvent):
            ms.rounds.append(RoundInfo(started_at=event.timestamp))
        elif isinstance(event, RoundEndEvent):
            if ms.rounds:
                ms.rounds[-1].ended_at = event.timestamp
        elif isinstance(event, RosterUpdate):
            self._handle_roster_update(event)
        elif isinstance(event, DeathEvent):
            if self._control_score_system and self._tick_loop:
                sim_time = self._tick_loop._current_tick / self._tick_loop.fps
                self._control_score_system.record_death(sim_time)

        self._ws_emit_match_state()

    def _handle_roster_update(self, event: RosterUpdate) -> None:
        """Process a single RosterUpdate into MatchState."""
        ms = self._match_state
        entry = event.entry
        if not entry.player_name:
            return  # empty slot (post-match clear or vacant)
        is_new = entry.player_name.upper() not in ms.players
        player = ms.get_or_create_player(entry.player_name)
        player.battletag = entry.battlenet_tag
        player.team = entry.team
        player.is_local = entry.is_local
        player.slot = event.slot

        if is_new:
            player.joined_at = event.timestamp
            local_tag = " [LOCAL]" if entry.is_local else ""
            _logger.info(f"Roster new player: {entry.player_name} ({entry.battlenet_tag}) "
                         f"team={entry.team} slot={event.slot}{local_tag}")

        # Resolve local team and derive team sides
        if entry.is_local and ms._local_team != entry.team:
            ms._local_team = entry.team
            ms.resolve_team_sides()
            _logger.info(f"Local team resolved: team={entry.team}")
            # Backfill detection: local player has zero stats but others don't
            if (not ms.is_backfill
                    and entry.kills == 0 and entry.deaths == 0
                    and entry.assists == 0 and entry.damage == 0
                    and entry.healed == 0 and entry.mitigated == 0):
                for other in ms.players.values():
                    if other.is_local or other.stats is None:
                        continue
                    s = other.stats
                    if (s.kills or s.deaths or s.assists
                            or s.damage or s.healing or s.mitigation):
                        ms.is_backfill = True
                        _logger.info("Backfill detected: other players have "
                                     "non-zero stats at match start")
                        break
        elif player.team is not None and ms._local_team is not None:
            from overwatchlooker.match_state import TeamSide
            player.team_side = (
                TeamSide.ALLY if player.team == ms._local_team
                else TeamSide.ENEMY
            )

        # Stats
        player.stats = StatsSnapshot(
            kills=entry.kills,
            deaths=entry.deaths,
            assists=entry.assists,
            damage=entry.damage,
            healing=entry.healed,
            mitigation=entry.mitigated,
        )

        # Hero swap detection
        if entry.hero_name and entry.hero_name != "UNKNOWN":
            from overwatchlooker.heroes import match_hero_name as _match_hero
            resolved = _match_hero(entry.hero_name) or entry.hero_name.title()
            current = player.current_hero
            if current is None or _edit_distance(resolved.lower(), current.lower()) > 2:
                player.hero_swaps.append(HeroSwap(
                    hero=resolved,
                    detected_at=event.timestamp,
                    source=HeroSource.OVERWOLF_ROSTER,
                    stats_at_detection=StatsSnapshot(
                        kills=entry.kills, deaths=entry.deaths,
                        assists=entry.assists, damage=entry.damage,
                        healing=entry.healed, mitigation=entry.mitigated,
                    ),
                ))
                if current is not None:
                    _logger.info(f"Roster hero swap: {entry.player_name} "
                                 f"{current} -> {entry.hero_name} "
                                 f"(K={entry.kills} D={entry.deaths} A={entry.assists})")

        # Role
        if entry.hero_role:
            try:
                player.role = PlayerRole(entry.hero_role)
            except ValueError:
                pass

        # Dump full state on first roster update with local player identified
        if is_new and entry.is_local:
            ms.dump_to_log("LocalPlayerRoster")

    _MATCH_END_DELAY = 5.0  # seconds to wait after match end before snapshotting

    def _trigger_match_end(self) -> None:
        """Schedule match end — waits 5s for final tab screenshots, then snapshots."""
        ms = self._match_state
        if ms._analysis_triggered:
            _logger.debug("Match end trigger skipped: already triggered")
            return
        # Skip if match state has no meaningful data (e.g. post-reset ghost events)
        if not ms.players and ms.started_at is None:
            _logger.debug("Match end trigger skipped: empty state")
            return
        # Debounce: ignore triggers within 5s of each other
        now = time.monotonic()
        if now - self._last_match_end_ts < 5.0:
            _logger.debug("Match end trigger skipped: debounce")
            return
        self._last_match_end_ts = now
        ms._analysis_triggered = True  # prevent re-trigger during delay
        self._pending_analysis.clear()  # mark analysis as pending

        _logger.info(f"Match end triggered: map={ms.map_name} result={ms.result} "
                     f"players={len(ms.players)} rounds={len(ms.rounds)} "
                     f"source={ms.result_source} (waiting {self._MATCH_END_DELAY}s for final tabs)")

        # Delay snapshot to allow final tab screenshots
        def _delayed_snapshot():
            time.sleep(self._MATCH_END_DELAY)
            self._finalize_match_end()

        threading.Thread(target=_delayed_snapshot, daemon=True).start()

    def _finalize_match_end(self) -> None:
        """Snapshot MatchState, analyze, print summary, reset for next match."""
        ms = self._match_state

        # Finalize control score — if neither team reached 2, infer from result
        if self._control_score_system and ms.control_score:
            last = ms.control_score[-1]
            if last[0] < 2 and last[1] < 2 and ms.result:
                if ms.result == MatchResult.VICTORY:
                    ms.control_score.append((2, last[1]))
                elif ms.result == MatchResult.DEFEAT:
                    ms.control_score.append((last[0], 2))

        _logger.info(f"Match end finalizing: hero_tabs={list(ms.hero_tabs.keys())}")
        ms.dump_to_log("MatchEnd")
        snapshot = ms.snapshot()

        # Start cooldown
        if self._tick_loop:
            fps = self._tick_loop.fps
            self._cooldown_until_tick = (
                self._tick_loop._current_tick + int(fps * _POST_SUBMIT_COOLDOWN)
            )

        # Reset subtitle/chat/team-side state
        if self._detector is not None:
            self._detector.reset_match()
        if self._chat_system is not None:
            self._chat_system.reset_match()
        if self._team_side_system is not None:
            self._team_side_system.reset_match()
        if self._control_score_system is not None:
            self._control_score_system.reset_match()

        # Reset match state for next match
        self._match_state = MatchState()

        # Schedule auto-recording stop
        self._auto_recording_schedule_stop()

        # Analyze, print, and submit in background thread
        def _finalize():
            self._analyze_snapshot(snapshot)
            summary = format_match_state(snapshot)
            from overwatchlooker.display import print_analysis
            print_analysis(summary)
            self._ws_emit({"type": "match_complete", "data": {
                "result": snapshot.result.value if snapshot.result else None,
                "map": snapshot.map_name,
                "mode": snapshot.mode,
                "duration_ms": snapshot.duration,
                "rank_min": snapshot.rank_min or None,
                "rank_max": snapshot.rank_max or None,
                "is_wide_match": snapshot.is_wide_match,
                "hero_bans": snapshot.hero_bans or None,
            }})
            show_notification("OverwatchLooker",
                              f"Match complete: {snapshot.result.value if snapshot.result else 'UNKNOWN'}")

            # Submit to MCP
            self._submit_to_mcp(snapshot)

            self._pending_analysis.set()  # signal analysis complete

        threading.Thread(target=_finalize, daemon=True).start()

    def _analyze_snapshot(self, snapshot: MatchState) -> None:
        """Run hero panel OCR and rank detection on the snapshot's tab screenshots."""
        try:
            import cv2
            import numpy as np
            from overwatchlooker.hero_panel import read_hero_panel, detect_rank_range, detect_hero_bans, detect_party_slots

            local = snapshot.local_player

            # OCR each hero's tab capture
            for hero_name, capture in snapshot.hero_tabs.items():
                img = cv2.imdecode(
                    np.frombuffer(capture.png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                panel_result = read_hero_panel(img)
                if panel_result and local:
                    # Find or create the HeroPanel entry for this hero
                    target_panel = None
                    for hp in local.hero_panels:
                        if hp.hero_name == hero_name:
                            target_panel = hp
                            break
                    if target_panel is None:
                        target_panel = HeroPanel(hero_name=hero_name, crop_png=b"")
                        local.hero_panels.append(target_panel)

                    target_panel.ocr_stats = [
                        {"label": s.label, "value": s.value,
                         "is_featured": s.is_featured}
                        for s in panel_result.stats
                    ]
                    _logger.info(f"Hero panel OCR: {len(panel_result.stats)} stats "
                                 f"for {hero_name}")

            # Rank detection (from latest tab screenshot)
            if snapshot.latest_tab:
                img = cv2.imdecode(
                    np.frombuffer(snapshot.latest_tab.png_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR)
                if img is not None:
                    rank = detect_rank_range(img)
                    if rank:
                        snapshot.rank_min = rank.min_rank
                        snapshot.rank_max = rank.max_rank
                        snapshot.is_wide_match = rank.is_wide
                        _logger.info(f"Rank: {rank.min_rank} - {rank.max_rank} "
                                     f"(wide={rank.is_wide})")

                # Hero bans (competitive only)
                from overwatchlooker.overwolf import GameType
                if snapshot.game_type == GameType.RANKED:
                    bans = detect_hero_bans(img)
                    if bans:
                        snapshot.hero_bans = bans

                # Party detection — map green indicators to ally players by role order
                assert img is not None
                party_flags = detect_party_slots(img)
                if party_flags:
                    from overwatchlooker.match_state import TeamSide
                    _ROLE_ORDER = {PlayerRole.TANK: 0, PlayerRole.DAMAGE: 1, PlayerRole.SUPPORT: 2}
                    allies = sorted(
                        [p for p in snapshot.players.values()
                         if p.team_side == TeamSide.ALLY],
                        key=lambda p: (_ROLE_ORDER.get(p.role, 99) if p.role else 99, p.player_name))
                    for i, flag in enumerate(party_flags):
                        if i < len(allies) and flag:
                            allies[i].in_party = True
                    party_names = [a.player_name for a in allies if a.in_party]
                    if party_names:
                        _logger.info(f"Party members: {party_names}")

        except Exception as e:
            _logger.warning(f"Snapshot analysis failed: {e}")

    _SKIP_GAME_TYPES = {GameType.PRACTICE, GameType.TUTORIAL, GameType.SKIRMISH,
                         GameType.HERO_MASTERY}

    def _submit_to_mcp(self, snapshot: MatchState) -> None:
        """Submit match to MCP server if enabled via --mcp flag."""
        if not self._use_mcp:
            return
        if snapshot.game_type in self._SKIP_GAME_TYPES:
            _logger.info(f"MCP: skipping {snapshot.game_type.value} match")
            return
        from overwatchlooker.config import MCP_URL
        if not MCP_URL:
            _logger.warning("--mcp flag set but MCP_URL not configured in .env")
            return
        try:
            from overwatchlooker.mcp_client import submit_match

            payload = build_mcp_payload(snapshot)

            # Attach latest tab screenshot
            png_bytes = None
            if snapshot.latest_tab:
                png_bytes = snapshot.latest_tab.png_bytes

            result = submit_match(payload, png_bytes=png_bytes)
            match_id = result.get("match_id")
            if match_id:
                _logger.info(f"MCP: submitted match {match_id}")
                self._ws_emit({"type": "mcp_submitted", "data": {"match_id": match_id}})
            else:
                _logger.info("MCP: submitted match (no ID returned)")
        except Exception as e:
            _logger.warning(f"MCP submission failed: {e}")

    def _auto_recording_start(self) -> None:
        """Start auto-recording if enabled and engine is available."""
        if not self._auto_recording or self._engine is None:
            return
        if self._auto_recording_active:
            _logger.debug("Auto-recording: already active, skipping start")
            return
        if self._recording:
            _logger.debug("Auto-recording: manual recording active, skipping")
            return
        # Cancel any pending stop timer (new match started before tail ended)
        if self._auto_stop_timer:
            self._auto_stop_timer.cancel()
            self._auto_stop_timer = None

        try:
            _AUTO_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            rec_dir = _AUTO_RECORDINGS_DIR / timestamp
            rec_dir.mkdir()
            base_path = rec_dir / "recording"
            info = self._engine.start_recording(str(base_path))
            self._auto_recording_active = True
            self._recording = True
            # Record Overwolf events alongside video
            if self._overwolf_system:
                from overwatchlooker.overwolf import OverwolfRecordingWriter
                frame_offset = self._tick_loop._current_tick if self._tick_loop else 0
                writer = OverwolfRecordingWriter(
                    rec_dir / "recording.overwolf.jsonl", frame_offset=frame_offset)
                self._overwolf_system.set_writer(writer)
                # Write current match state so replay has the full context
                self._write_match_state_to_recording(writer, frame_offset)
            _logger.info(f"Auto-recording started: {info.video_path}")
            print_status(f"Auto-recording to {info.video_path}")
            self._ws_emit({"type": "state", "recording": True})
            self._rebuild_menu()
        except Exception as e:
            _logger.warning(f"Auto-recording start failed: {e}")

    def _write_match_state_to_recording(self, writer, frame_offset: int) -> None:
        """Write synthetic events for the current match state into the recording.

        This ensures replay has the MatchStartEvent and all context that was
        already processed before the recording started.
        """
        from overwatchlooker.overwolf import (
            GameModeUpdate, GameTypeUpdate, MapUpdate, MatchStartEvent,
            PseudoMatchIdUpdate, QueueTypeUpdate, RosterEntry, RosterUpdate,
            RoundStartEvent,
        )
        ms = self._match_state
        ts = ms.started_at or 0
        frame = frame_offset  # will be written as frame 0 after offset subtraction

        writer.write(MatchStartEvent(timestamp=ts), frame)

        if ms.map_name and ms.map_code:
            writer.write(MapUpdate(code=ms.map_code, name=ms.map_name, timestamp=ts), frame)
        if ms.mode and ms.mode_code:
            writer.write(GameModeUpdate(code=ms.mode_code, name=ms.mode, timestamp=ts), frame)
        if ms.game_type:
            writer.write(GameTypeUpdate(game_type=ms.game_type, timestamp=ts), frame)
        if ms.queue_type:
            writer.write(QueueTypeUpdate(queue_type=ms.queue_type, timestamp=ts), frame)
        if ms.pseudo_match_id:
            writer.write(PseudoMatchIdUpdate(
                pseudo_match_id=ms.pseudo_match_id, timestamp=ts), frame)
        for r in ms.rounds:
            writer.write(RoundStartEvent(timestamp=r.started_at), frame)

        # Write current roster
        for p in ms.players.values():
            if p.slot is not None:
                entry = RosterEntry(
                    player_name=p.player_name,
                    battlenet_tag=p.battletag,
                    is_local=p.is_local,
                    is_teammate=(p.team == ms._local_team) if ms._local_team is not None else False,
                    hero_name=p.current_hero or "",
                    hero_role=p.role.value if p.role else "",
                    team=p.team if p.team is not None else 0,
                    kills=p.stats.kills if p.stats else 0,
                    deaths=p.stats.deaths if p.stats else 0,
                    assists=p.stats.assists if p.stats else 0,
                    damage=p.stats.damage if p.stats else 0.0,
                    healed=p.stats.healing if p.stats else 0.0,
                    mitigated=p.stats.mitigation if p.stats else 0.0,
                )
                writer.write(RosterUpdate(slot=p.slot, entry=entry, timestamp=ts), frame)

    def _auto_recording_schedule_stop(self) -> None:
        """Schedule auto-recording stop after the configured tail duration."""
        if not self._auto_recording_active:
            return
        _logger.info(f"Auto-recording: scheduling stop in {self._auto_recording_tail}s")

        def _stop():
            self._auto_recording_stop()

        self._auto_stop_timer = threading.Timer(self._auto_recording_tail, _stop)
        self._auto_stop_timer.daemon = True
        self._auto_stop_timer.start()

    def _auto_recording_stop(self) -> None:
        """Stop auto-recording."""
        if not self._auto_recording_active or self._engine is None:
            return
        try:
            self._engine.stop_recording()
            self._auto_recording_active = False
            self._recording = False
            if self._overwolf_system:
                self._overwolf_system.clear_writer()
            _logger.info("Auto-recording stopped")
            print_status("Auto-recording saved.")
            show_notification("OverwatchLooker", "Auto-recording saved.")
            self._ws_emit({"type": "state", "recording": False})
            self._rebuild_menu()
        except Exception as e:
            _logger.warning(f"Auto-recording stop failed: {e}")

    def store_valid_tab(self, png_bytes: bytes, timestamp: float, filename: str,
                        tick: int = 0) -> None:
        """Store a valid Tab screenshot (called by TabCaptureSystem).

        timestamp is time.monotonic() in live mode, sim_time in replay mode.
        Stores per-hero (if panel visible) + latest raw for rank detection.
        """
        if self._in_post_detection_cooldown():
            _logger.debug(f"Tab ignored (post-detection cooldown): {filename}")
            return
        with self._lock:
            ms = self._match_state

            # Always keep the latest tab for rank detection
            ms.latest_tab = TabScreenshot(
                png_bytes=png_bytes, sim_time=timestamp, filename=filename)

            # Check if hero panel is visible and store per-hero
            local = ms.local_player
            if local and local.current_hero:
                hero = local.current_hero
                existing = ms.hero_tabs.get(hero)
                if existing is None or tick > existing.tick:
                    # Only store if the screenshot has a hero panel
                    from overwatchlooker.hero_panel import _detect_panel
                    import cv2
                    import numpy as np
                    img = cv2.imdecode(
                        np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None and _detect_panel(img) is not None:
                        from overwatchlooker.match_state import HeroTabCapture
                        ms.hero_tabs[hero] = HeroTabCapture(
                            hero_name=hero, png_bytes=png_bytes,
                            tick=tick, filename=filename)
                        _logger.info(f"Tab stored for {hero}: {filename}")
                    else:
                        _logger.debug(f"Tab has no hero panel, skipped for {hero}")

        print_status(f"Tab screenshot saved: {filename}")
        self._ws_emit({"type": "tab_capture", "filename": filename,
                       "timestamp": timestamp})

        # Run rank + ban detection early if not yet populated (competitive only)
        self._try_early_detection(png_bytes)

    def _try_early_detection(self, png_bytes: bytes) -> None:
        """Run rank + ban detection on a tab screenshot if not yet populated."""
        ms = self._match_state
        from overwatchlooker.overwolf import GameType
        is_ranked = ms.game_type == GameType.RANKED
        needs_rank = not ms.rank_min
        needs_bans = is_ranked and not ms.hero_bans

        if not needs_rank and not needs_bans:
            return

        def _detect():
            try:
                import cv2
                import numpy as np
                from overwatchlooker.hero_panel import detect_rank_range, detect_hero_bans

                img = cv2.imdecode(
                    np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    return

                if needs_rank:
                    rank = detect_rank_range(img)
                    if rank:
                        ms.rank_min = rank.min_rank
                        ms.rank_max = rank.max_rank
                        ms.is_wide_match = rank.is_wide
                        _logger.info(f"Early rank: {rank.min_rank} - {rank.max_rank} "
                                     f"(wide={rank.is_wide})")
                        self._ws_emit_match_state()

                if needs_bans:
                    bans = detect_hero_bans(img)
                    if bans:
                        ms.hero_bans = bans
                        _logger.info(f"Early bans: {bans}")
                        self._ws_emit_match_state()
            except Exception as e:
                _logger.debug(f"Early detection failed: {e}")

        threading.Thread(target=_detect, daemon=True).start()

    def store_hero_crop(self, name: str, crop: bytes) -> None:
        """Store a hero panel crop into local player's hero_panels."""
        if self._in_post_detection_cooldown():
            return
        with self._lock:
            ms = self._match_state
            local = ms.local_player
            if local is None:
                # No local player known yet — create a placeholder
                # (will be resolved when roster arrives)
                all_panels = [hp for p in ms.players.values() for hp in p.hero_panels]
                if any(_edit_distance(name.lower(), hp.hero_name.lower()) <= 2
                       for hp in all_panels):
                    _logger.debug(f"Hero crop dedup skip: {name}")
                    return
                # Store on a temporary _LOCAL_ player
                player = ms.get_or_create_player("_LOCAL_")
                player.hero_panels.append(HeroPanel(hero_name=name, crop_png=crop))
            else:
                if any(_edit_distance(name.lower(), hp.hero_name.lower()) <= 2
                       for hp in local.hero_panels):
                    _logger.debug(f"Hero crop dedup skip: {name}")
                    return
                local.hero_panels.append(HeroPanel(hero_name=name, crop_png=crop))
            _logger.info(f"Stored hero crop: {name}")
            self._ws_emit({"type": "hero_crop", "name": name})

    def _on_control_score_change(self, transitions: list[tuple[int, int]]) -> None:
        """Callback when control score changes."""
        self._match_state.control_score = transitions
        self._ws_emit_match_state()

    def _on_team_side_detected(self, side: str) -> None:
        """Callback when ATTACK/DEFEND label is detected from frame OCR."""
        self._match_state.initial_team_side = side
        self._ws_emit_match_state()

    def _on_hero_switch(self, player: str, hero: str, sim_time: float) -> None:
        """Callback when a hero switch is detected in subtitles."""
        ms = self._match_state
        p = ms.get_or_create_player(player)
        current = p.current_hero
        if current is None or _edit_distance(hero.lower(), current.lower()) > 2:
            p.hero_swaps.append(HeroSwap(
                hero=hero,
                detected_at=int(sim_time * 1000),
                source=HeroSource.SUBTITLE_OCR,
            ))
            _logger.info(f"Subtitle hero switch: {player} -> {hero} (t={sim_time:.1f}s)")
        else:
            _logger.debug(f"Subtitle hero switch deduped: {player} -> {hero} (current={current})")
        self._ws_emit({"type": "hero_switch", "player": player,
                       "hero": hero, "time": sim_time})
        self._ws_emit_match_state()

    def _on_player_change(self, player: str, event: str, sim_time: float) -> None:
        """Callback when a player joins or leaves the game."""
        _logger.info(f"Chat player change: {player} {event} (t={sim_time:.1f}s)")
        ms = self._match_state
        p = ms.get_or_create_player(player)
        epoch_ms = int(sim_time * 1000)
        if event == "joined":
            p.joined_at = epoch_ms
        elif event == "left":
            p.left_at = epoch_ms
        self._ws_emit({"type": "player_change", "player": player,
                       "event": event, "time": sim_time})
        self._ws_emit_match_state()

    def _on_detected(self, result: str, detection_time: float = 0.0) -> None:
        """Immediate callback when VICTORY/DEFEAT is first detected."""
        _logger.info(f"Subtitle detected: {result} (t={detection_time:.1f}s)")
        self._ws_emit({"type": "detection", "result": result, "time": detection_time})
        show_notification("OverwatchLooker", f"{result} detected!")
        print_status(f"Detected: {result}.")

    def _on_detection(self, result: str, detection_time: float = 0.0) -> None:
        """Delayed callback: subtitle OCR detected VICTORY/DEFEAT — trigger match end."""
        _logger.info(f"Subtitle detection trigger: {result} (t={detection_time:.1f}s)")
        self._match_state.dump_to_log("SubtitleDetection")
        ms = self._match_state
        # Set result from subtitle if Overwolf hasn't already
        if ms.result is None:
            if result == "VICTORY":
                ms.result = MatchResult.VICTORY
            elif result == "DEFEAT":
                ms.result = MatchResult.DEFEAT
            ms.result_source = ResultSource.SUBTITLE
        if ms.ended_at is None:
            ms.ended_at = int(detection_time * 1000)
        self._ws_emit_match_state()
        self._trigger_match_end()

    # ------------------------------------------------------------------
    # Engine lifecycle: polling for Overwatch window
    # ------------------------------------------------------------------

    def _is_overwatch_running(self) -> bool:
        """Check if any window belongs to overwatch.exe."""
        import ctypes
        import ctypes.wintypes

        # EnumWindows approach: check all top-level windows for overwatch.exe
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        found = [False]

        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def enum_callback(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
            if not handle:
                return True
            try:
                buf = ctypes.create_unicode_buffer(260)
                size = ctypes.wintypes.DWORD(260)
                if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    exe = buf.value.rsplit("\\", 1)[-1].lower()
                    if exe == "overwatch.exe":
                        found[0] = True
                        return False  # stop enumeration
            finally:
                kernel32.CloseHandle(handle)
            return True

        user32.EnumWindows(enum_callback, 0)
        return found[0]

    def _poll_for_overwatch(self) -> None:
        """Polling thread: waits for OW window, creates engine, starts tick loop."""
        stable_since: float | None = None

        while not self._poll_stop.is_set():
            if self._engine is not None:
                # Engine exists — monitor health
                self._monitor_engine_health()
                self._poll_stop.wait(_OW_POLL_INTERVAL)
                continue

            ow_present = self._is_overwatch_running()

            if ow_present:
                if stable_since is None:
                    stable_since = time.monotonic()
                    _logger.info("Overwatch window detected, waiting for stability...")
                elif time.monotonic() - stable_since >= _OW_STABILITY_TIME:
                    # Stable long enough — create engine and start
                    try:
                        self._create_engine_and_start()
                        stable_since = None
                    except Exception as e:
                        _logger.error(f"Failed to create capture engine: {e}")
                        stable_since = None
            else:
                if stable_since is not None:
                    _logger.info("Overwatch window disappeared during stability wait")
                    stable_since = None

            self._poll_stop.wait(_OW_POLL_INTERVAL)

    def _create_engine_and_start(self) -> None:
        """Create a memoir CaptureEngine targeting Overwatch and start the tick loop."""
        from memoir_capture import CaptureEngine, MetaKeyEntry, WindowExeTarget

        fps = 10
        key_map = [
            MetaKeyEntry(bit_index=0, virtual_key=0x09, name="tab"),
            MetaKeyEntry(bit_index=1, virtual_key=0xA4, name="alt_l"),
            MetaKeyEntry(bit_index=2, virtual_key=0xA5, name="alt_r"),
        ]
        self._engine = CaptureEngine(
            WindowExeTarget("(?i)overwatch"),
            max_fps=fps,
            key_map=key_map,
            record_gop=20,
        )
        self._engine.start()

        from overwatchlooker.tick import (
            ChatSystem, ControlScoreSystem, MemoirFrameSource, MemoirInputSource,
            OverwolfSystem, SubtitleSystem, TabCaptureSystem, TeamSideSystem,
            TickLoop,
        )
        from overwatchlooker.config import SUBTITLE_POLL_INTERVAL

        frame_source = MemoirFrameSource(self._engine, fps)
        input_source = MemoirInputSource(frame_source, [
            {"bit_index": k.bit_index, "name": k.name} for k in key_map
        ])

        self._tick_loop = TickLoop(fps, frame_source, input_source)

        tab_system = TabCaptureSystem(self, fps=fps)
        self._tick_loop.register(tab_system.on_tick, every_n_ticks=1)

        team_side = TeamSideSystem(self, on_detected=self._on_team_side_detected)
        self._tick_loop.register(team_side.on_tick, every_n_ticks=3)
        self._team_side_system = team_side

        control_score = ControlScoreSystem(on_score_change=self._on_control_score_change)
        self._tick_loop.register(control_score.on_tick, every_n_ticks=int(fps * 2))
        self._control_score_system = control_score

        # Subtitle + chat OCR only when Overwolf is not connected
        if not self._overwolf_queue:
            subtitle_interval = max(1, int(fps * SUBTITLE_POLL_INTERVAL))
            subtitle_system = SubtitleSystem(on_match=self._on_detection,
                                             on_detected=self._on_detected,
                                             on_hero_switch=self._on_hero_switch,
                                             transcript=self._use_transcript,
                                             detection_delay_ticks=0)
            self._tick_loop.register(subtitle_system.on_tick, every_n_ticks=subtitle_interval)
            self._detector = subtitle_system
            self._subtitle_system = subtitle_system

            chat_system = ChatSystem(on_player_change=self._on_player_change)
            self._tick_loop.register(chat_system.on_tick, every_n_ticks=subtitle_interval)
            self._chat_system = chat_system

        # Overwolf event system (drains queue each tick)
        if self._overwolf_queue:
            overwolf_system = OverwolfSystem(
                self._overwolf_queue, on_event=self._on_overwolf_event)
            self._tick_loop.register(overwolf_system.on_tick, every_n_ticks=1)
            self._overwolf_system = overwolf_system

        tick_thread = threading.Thread(target=self._tick_loop.run, daemon=True)
        tick_thread.start()

        print_status("Overwatch detected — capture engine started.")
        self._ws_emit({"type": "state", "active": True})

    def _monitor_engine_health(self) -> None:
        """Check if the engine has faulted (e.g. OW closed)."""
        if self._engine is None:
            return
        err = self._engine.get_last_error()
        if err:
            _logger.warning(f"Capture engine error: {err}")
            self._tear_down_engine()
            print_status("Overwatch closed. Waiting for reconnect...")
            self._ws_emit({"type": "state", "active": True})
            if self._recording:
                self._recording = False
                self._ws_emit({"type": "state", "recording": False})

    def _tear_down_engine(self) -> None:
        """Stop tick loop and engine, but keep polling alive."""
        if self._tick_loop:
            self._tick_loop.stop()
            self._tick_loop = None
        if hasattr(self, '_subtitle_system') and self._subtitle_system:
            self._subtitle_system.close()
            self._subtitle_system = None
        if self._overwolf_system:
            self._overwolf_system.clear_writer()
            self._overwolf_system = None
        self._chat_system = None
        self._detector = None
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def _start_listening(self) -> None:
        if self._active:
            return

        self._active = True

        if self._replay_source:
            # Replay mode: use ReplayFrameSource/ReplayInputSource directly
            from overwatchlooker.tick import (
                ChatSystem, ControlScoreSystem, OverwolfSystem,
                ReplayFrameSource, ReplayInputSource,
                ReplayOverwolfSource, SubtitleSystem, TabCaptureSystem,
                TeamSideSystem, TickLoop,
            )
            from overwatchlooker.overwolf import OverwolfEventQueue, load_overwolf_events
            from overwatchlooker.config import SUBTITLE_POLL_INTERVAL

            fps = self._replay_source.fps
            frame_source = ReplayFrameSource(self._replay_source.reader)
            input_source = ReplayInputSource(self._replay_source.events)

            self._tick_loop = TickLoop(fps, frame_source, input_source)

            tab_system = TabCaptureSystem(self, fps=fps)
            self._tick_loop.register(tab_system.on_tick, every_n_ticks=1)

            team_side = TeamSideSystem(self, on_detected=self._on_team_side_detected)
            self._tick_loop.register(team_side.on_tick, every_n_ticks=3)
            self._team_side_system = team_side

            control_score = ControlScoreSystem(on_score_change=self._on_control_score_change)
            self._tick_loop.register(control_score.on_tick, every_n_ticks=int(fps * 2))
            self._control_score_system = control_score

            # Replay Overwolf events if recording has them
            overwolf_events_path = self._replay_source.overwolf_events_path
            has_overwolf_replay = overwolf_events_path and overwolf_events_path.exists()
            # Subtitle + chat OCR only when no Overwolf events in recording
            if not has_overwolf_replay:
                subtitle_interval = max(1, int(fps * SUBTITLE_POLL_INTERVAL))
                subtitle_system = SubtitleSystem(on_match=self._on_detection,
                                                 on_detected=self._on_detected,
                                                 on_hero_switch=self._on_hero_switch,
                                                 transcript=self._use_transcript,
                                                 detection_delay_ticks=0)
                self._tick_loop.register(subtitle_system.on_tick, every_n_ticks=subtitle_interval)
                self._detector = subtitle_system
                self._subtitle_system = subtitle_system

                chat_system = ChatSystem(on_player_change=self._on_player_change)
                self._tick_loop.register(chat_system.on_tick, every_n_ticks=subtitle_interval)
                self._chat_system = chat_system

            if has_overwolf_replay:
                replay_queue = OverwolfEventQueue()
                recorded_events = load_overwolf_events(overwolf_events_path)
                replay_source_ow = ReplayOverwolfSource(recorded_events, replay_queue)
                self._tick_loop.register_pre_tick(replay_source_ow.advance_to)
                overwolf_system = OverwolfSystem(
                    replay_queue, on_event=self._on_overwolf_event)
                self._tick_loop.register(overwolf_system.on_tick, every_n_ticks=1)
                self._overwolf_system = overwolf_system
                _logger.info(f"Replaying {len(recorded_events)} Overwolf events")

            detect_mode = "replay"
        else:
            # Live mode: start polling thread for OW window
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_for_overwatch, daemon=True
            )
            self._poll_thread.start()
            detect_mode = "subtitle"

        print_status(f"Listening (detection={detect_mode}). Tab=screenshot.")
        self._ws_emit({"type": "state", "active": True})

    def _stop_listening(self) -> None:
        if not self._active:
            return
        self._active = False

        # Stop polling thread
        self._poll_stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None

        # Stop engine and tick loop
        self._tear_down_engine()

        # Also stop tick loop if it was set up for replay
        if self._tick_loop:
            self._tick_loop.stop()
            self._tick_loop = None

        print_status("Stopped listening.")
        self._ws_emit({"type": "state", "active": False})

    def _on_submit_tab(self, result: str) -> None:
        """Manually submit with a given result — set result and trigger match end."""
        _logger.info(f"Manual submit: {result}")
        ms = self._match_state
        if result == "VICTORY":
            ms.result = MatchResult.VICTORY
        elif result == "DEFEAT":
            ms.result = MatchResult.DEFEAT
        ms.result_source = ResultSource.SUBTITLE
        if ms.ended_at is None:
            now_ms = int(time.time() * 1000)
            ms.ended_at = now_ms
        self._trigger_match_end()

    def _on_submit_win(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_submit_tab("VICTORY")

    def _on_submit_loss(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_submit_tab("DEFEAT")

    def _on_toggle_recording(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Toggle recording on/off via the memoir engine."""
        if self._engine is None:
            print_error("No capture engine running. Start Overwatch first.")
            return

        if self._auto_recording_active:
            print_error("Auto-recording is active. Stop it via match end or restart without --auto-recording.")
            return

        if self._recording:
            try:
                self._engine.stop_recording()
                self._recording = False
                if self._overwolf_system:
                    self._overwolf_system.clear_writer()
                print_status("Recording stopped.")
                show_notification("OverwatchLooker", "Recording saved.")
                self._ws_emit({"type": "state", "recording": False})
            except Exception as e:
                print_error(f"Failed to stop recording: {e}")
        else:
            try:
                _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                rec_dir = _RECORDINGS_DIR / timestamp
                rec_dir.mkdir()
                base_path = rec_dir / "recording"
                info = self._engine.start_recording(str(base_path))
                self._recording = True
                # Start Overwolf event recording alongside video
                if self._overwolf_system:
                    from overwatchlooker.overwolf import OverwolfRecordingWriter
                    frame_offset = self._tick_loop._current_tick if self._tick_loop else 0
                    writer = OverwolfRecordingWriter(
                        rec_dir / "recording.overwolf.jsonl", frame_offset=frame_offset)
                    self._overwolf_system.set_writer(writer)
                print_status(f"Recording to {info.video_path}")
                show_notification("OverwatchLooker", "Recording started.")
                self._ws_emit({"type": "state", "recording": True})
            except Exception as e:
                print_error(f"Failed to start recording: {e}")
        self._rebuild_menu()

    def wait_for_analysis(self, timeout: float = 30.0) -> None:
        """Block until any pending match analysis completes."""
        self._pending_analysis.wait(timeout)

    def _on_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        if self._auto_stop_timer:
            self._auto_stop_timer.cancel()
            self._auto_stop_timer = None
        if self._engine and self._recording:
            try:
                self._engine.stop_recording()
                self._recording = False
                self._auto_recording_active = False
            except Exception:
                pass
        self._stop_listening()
        icon.stop()

    def _on_toggle(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        try:
            _logger.debug(f"Toggle clicked, active={self._active}")
            if self._active:
                self._stop_listening()
            else:
                self._start_listening()
            _logger.debug(f"After toggle, active={self._active}")
            self._rebuild_menu()
        except Exception:
            _logger.error(f"Toggle error:\n{traceback.format_exc()}")

    def _rebuild_menu(self) -> None:
        """Rebuild the tray menu to reflect current state."""
        if self._icon is None:
            return
        label = "Stop Listening" if self._active else "Start Listening"
        rec_label = "Stop Recording" if self._recording else "Start Recording"
        self._icon.menu = pystray.Menu(
            pystray.MenuItem(label, self._on_toggle, default=True),
            pystray.MenuItem(rec_label, self._on_toggle_recording),
            pystray.MenuItem("Submit last tab (win)", self._on_submit_win),
            pystray.MenuItem("Submit last tab (loss)", self._on_submit_loss),
            pystray.MenuItem("Quit", self._on_quit),
        )
        self._icon.update_menu()

    def run(self) -> None:
        """Main entry point -- blocks on the tray icon run loop."""
        self._icon = pystray.Icon(
            name="OverwatchLooker",
            icon=_create_icon_image(),
            title="OverwatchLooker",
            menu=pystray.Menu(
                pystray.MenuItem("Stop Listening", self._on_toggle, default=True),
                pystray.MenuItem("Start Recording", self._on_toggle_recording),
                pystray.MenuItem("Submit last tab (win)", self._on_submit_win),
                pystray.MenuItem("Submit last tab (loss)", self._on_submit_loss),
                pystray.MenuItem("Quit", self._on_quit),
            ),
        )

        def setup(icon: pystray.Icon):
            try:
                icon.visible = True
                self._start_listening()
                _logger.debug(f"Setup complete, active={self._active}")
            except Exception:
                _logger.error(f"Setup error:\n{traceback.format_exc()}")

        print_status("OverwatchLooker starting. Check system tray. Press Ctrl+C to quit.")

        # Run tray icon in a daemon thread so the main thread can catch Ctrl+C
        tray_thread = threading.Thread(
            target=self._icon.run, kwargs={"setup": setup}, daemon=True
        )
        tray_thread.start()

        try:
            while tray_thread.is_alive():
                tray_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            _logger.info("KeyboardInterrupt received")
            self._shutdown()

        if not tray_thread.is_alive():
            _logger.warning("Tray icon thread exited unexpectedly")
            self._shutdown()

    def _shutdown(self) -> None:
        print_status("Shutting down...")
        if self._engine and self._recording:
            try:
                self._engine.stop_recording()
                self._recording = False
                print_status("Recording saved on shutdown.")
            except Exception:
                pass
        self._stop_listening()
        if self._overwolf:
            self._overwolf.stop()
        if self._icon:
            self._icon.stop()
