"""Tests for Overwolf GEP WebSocket receiver, event parsing, queue, and serialization."""

import json

import pytest

from overwatchlooker.overwolf import (
    AssistEvent,
    AssistsUpdate,
    BattleTagUpdate,
    DeathEvent,
    DeathsUpdate,
    EliminationEvent,
    EliminationsUpdate,
    GameModeUpdate,
    GameState,
    GameStateUpdate,
    GameStatusEvent,
    GameType,
    GameTypeUpdate,
    MapUpdate,
    MatchEndEvent,
    MatchOutcome,
    MatchOutcomeUpdate,
    MatchStartEvent,
    ObjectiveKillsUpdate,
    OverwolfEventQueue,
    OverwolfRecordingWriter,
    PseudoMatchIdUpdate,
    QueueType,
    QueueTypeUpdate,
    RespawnEvent,
    ReviveEvent,
    RosterEntry,
    RosterUpdate,
    RoundEndEvent,
    RoundStartEvent,
    _parse_message,
    _safe_int,
    deserialize_event,
    load_overwolf_events,
    serialize_event,
)


# ---------------------------------------------------------------------------
# _safe_int
# ---------------------------------------------------------------------------

class TestSafeInt:
    def test_string_number(self):
        assert _safe_int("42") == 42

    def test_none(self):
        assert _safe_int(None) == 0

    def test_empty_string(self):
        assert _safe_int("") == 0

    def test_actual_int(self):
        assert _safe_int(7) == 7


# ---------------------------------------------------------------------------
# RosterEntry.from_json
# ---------------------------------------------------------------------------

class TestRosterEntry:
    def test_full_entry(self):
        raw = json.dumps({
            "player_name": "EDGED217", "battlenet_tag": "EDGED217#2842",
            "is_local": False, "is_teammate": True,
            "hero_name": "MOIRA", "hero_role": "SUPPORT",
            "team": 0, "kills": 4, "deaths": 0, "assists": 0,
            "damage": 462.16, "healed": 274.866, "mitigated": 0,
        })
        entry = RosterEntry.from_json(raw)
        assert entry.player_name == "EDGED217"
        assert entry.battlenet_tag == "EDGED217#2842"
        assert entry.is_local is False
        assert entry.is_teammate is True
        assert entry.hero_name == "MOIRA"
        assert entry.damage == 462.16
        assert entry.healed == 274.866

    def test_missing_fields_default(self):
        entry = RosterEntry.from_json("{}")
        assert entry.player_name == ""
        assert entry.kills == 0
        assert entry.damage == 0.0

    def test_null_stat_fields(self):
        """Overwolf sometimes sends null for stat fields mid-match."""
        raw = json.dumps({
            "player_name": "TEST", "battlenet_tag": "Test#1",
            "is_local": False, "is_teammate": True,
            "hero_name": None, "hero_role": None,
            "team": 1, "kills": None, "deaths": None, "assists": None,
            "damage": None, "healed": None, "mitigated": None,
        })
        entry = RosterEntry.from_json(raw)
        assert entry.hero_name == ""
        assert entry.hero_role == ""
        assert entry.kills == 0
        assert entry.deaths == 0
        assert entry.damage == 0.0
        assert entry.healed == 0.0


# ---------------------------------------------------------------------------
# _parse_message — game_status
# ---------------------------------------------------------------------------

class TestParseGameStatus:
    def test_running_true(self):
        events = _parse_message({"type": "game_status", "data": {"running": True}, "timestamp": 100})
        assert len(events) == 1
        assert isinstance(events[0], GameStatusEvent)
        assert events[0].running is True
        assert events[0].timestamp == 100

    def test_running_false(self):
        events = _parse_message({"type": "game_status", "data": {"running": False}, "timestamp": 200})
        assert events[0].running is False


# ---------------------------------------------------------------------------
# _parse_message — discrete events
# ---------------------------------------------------------------------------

class TestParseDiscreteEvents:
    def test_elimination(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "elimination", "data": "8"}
        ]}, "timestamp": 100})
        assert len(events) == 1
        assert isinstance(events[0], EliminationEvent)
        assert events[0].total == 8

    def test_death(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "death", "data": "3"}
        ]}, "timestamp": 100})
        assert isinstance(events[0], DeathEvent)
        assert events[0].total == 3

    def test_assist(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "assist", "data": "5"}
        ]}, "timestamp": 100})
        assert isinstance(events[0], AssistEvent)

    def test_match_start(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "match_start", "data": ""}
        ]}, "timestamp": 100})
        assert isinstance(events[0], MatchStartEvent)

    def test_match_end(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "match_end", "data": ""}
        ]}, "timestamp": 100})
        assert isinstance(events[0], MatchEndEvent)

    def test_round_start(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "round_start", "data": ""}
        ]}, "timestamp": 100})
        assert isinstance(events[0], RoundStartEvent)

    def test_round_end(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "round_end", "data": ""}
        ]}, "timestamp": 100})
        assert isinstance(events[0], RoundEndEvent)

    def test_respawn(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "respawn", "data": None}
        ]}, "timestamp": 100})
        assert isinstance(events[0], RespawnEvent)

    def test_revive(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "revive", "data": None}
        ]}, "timestamp": 100})
        assert isinstance(events[0], ReviveEvent)

    def test_multiple_events_in_one_message(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "elimination", "data": "1"},
            {"name": "death", "data": "1"},
        ]}, "timestamp": 100})
        assert len(events) == 2
        assert isinstance(events[0], EliminationEvent)
        assert isinstance(events[1], DeathEvent)

    def test_unknown_event_ignored(self):
        events = _parse_message({"type": "event", "data": {"events": [
            {"name": "unknown_event", "data": ""}
        ]}, "timestamp": 100})
        assert len(events) == 0


# ---------------------------------------------------------------------------
# _parse_message — info updates
# ---------------------------------------------------------------------------

class TestParseInfoUpdates:
    def test_game_state(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"game_info": {"game_state": "match_in_progress"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert isinstance(events[0], GameStateUpdate)
        assert events[0].state == GameState.MATCH_IN_PROGRESS

    def test_game_mode(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"game_info": {"game_mode": "0023"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert isinstance(events[0], GameModeUpdate)
        assert events[0].code == "0023"
        assert events[0].name == "Control"

    def test_game_mode_push(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"game_info": {"game_mode": "64"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert events[0].name == "Push"

    def test_game_mode_without_leading_zeros(self):
        """Overwolf sometimes sends mode code without leading zeros."""
        events = _parse_message({"type": "info_update", "data": {
            "info": {"game_info": {"game_mode": "21"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert isinstance(events[0], GameModeUpdate)
        assert events[0].code == "21"
        assert events[0].name == "Escort"

    def test_game_mode_unknown(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"game_info": {"game_mode": "9999"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert events[0].name == "Unknown (9999)"

    def test_battle_tag(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"game_info": {"battle_tag": "Player#1234"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert isinstance(events[0], BattleTagUpdate)
        assert events[0].battle_tag == "Player#1234"

    def test_game_type(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"match_info": {"game_type": "RANKED"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert isinstance(events[0], GameTypeUpdate)
        assert events[0].game_type == GameType.RANKED

    def test_queue_type(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"match_info": {"game_queue_type": "ROLE_QUEUE"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert isinstance(events[0], QueueTypeUpdate)
        assert events[0].queue_type == QueueType.ROLE_QUEUE

    def test_map(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"match_info": {"map": "212"}},
            "feature": "match_info"
        }, "timestamp": 100})
        assert isinstance(events[0], MapUpdate)
        assert events[0].name == "King's Row"

    def test_pseudo_match_id(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"match_info": {"pseudo_match_id": "abc-123"}},
            "feature": "match_info"
        }, "timestamp": 100})
        assert isinstance(events[0], PseudoMatchIdUpdate)
        assert events[0].pseudo_match_id == "abc-123"

    def test_match_outcome(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"match_info": {"match_outcome": "victory"}},
            "feature": "match_info"
        }, "timestamp": 100})
        assert isinstance(events[0], MatchOutcomeUpdate)
        assert events[0].outcome == MatchOutcome.VICTORY

    def test_eliminations(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"kill": {"eliminations": "6"}},
            "feature": "kill"
        }, "timestamp": 100})
        assert isinstance(events[0], EliminationsUpdate)
        assert events[0].count == 6

    def test_objective_kills(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"kill": {"objective_kills": "2"}},
            "feature": "kill"
        }, "timestamp": 100})
        assert isinstance(events[0], ObjectiveKillsUpdate)

    def test_deaths_info(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"death": {"deaths": "5"}},
            "feature": "death"
        }, "timestamp": 100})
        assert isinstance(events[0], DeathsUpdate)

    def test_assists_info(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"assist": {"assist": "3"}},
            "feature": "assist"
        }, "timestamp": 100})
        assert isinstance(events[0], AssistsUpdate)

    def test_roster(self):
        roster_json = json.dumps({
            "player_name": "TEST", "battlenet_tag": "TEST#1234",
            "is_local": True, "is_teammate": True,
            "hero_name": "ANA", "hero_role": "SUPPORT",
            "team": 0, "kills": 5, "deaths": 1, "assists": 3,
            "damage": 1000, "healed": 5000, "mitigated": 0,
        })
        events = _parse_message({"type": "info_update", "data": {
            "info": {"roster": {"roster_3": roster_json}},
            "feature": "roster"
        }, "timestamp": 100})
        assert len(events) == 1
        assert isinstance(events[0], RosterUpdate)
        assert events[0].slot == 3
        assert events[0].entry.player_name == "TEST"

    def test_multiple_info_fields(self):
        """Multiple fields in one info_update produce multiple events."""
        events = _parse_message({"type": "info_update", "data": {
            "info": {
                "game_info": {"game_state": "match_in_progress", "battle_tag": "P#1"},
            },
            "feature": "game_info"
        }, "timestamp": 100})
        assert len(events) == 2

    def test_unknown_type_ignored(self):
        events = _parse_message({"type": "unknown_type", "data": {}, "timestamp": 100})
        assert len(events) == 0

    def test_unknown_game_state_ignored(self):
        events = _parse_message({"type": "info_update", "data": {
            "info": {"game_info": {"game_state": "nonexistent_state"}},
            "feature": "game_info"
        }, "timestamp": 100})
        assert len(events) == 0


# ---------------------------------------------------------------------------
# OverwolfEventQueue
# ---------------------------------------------------------------------------

class TestOverwolfEventQueue:
    def test_push_and_drain(self):
        q = OverwolfEventQueue()
        q.push(MatchStartEvent(timestamp=100))
        q.push(MatchEndEvent(timestamp=200))
        events = q.drain()
        assert len(events) == 2
        assert isinstance(events[0], MatchStartEvent)
        assert isinstance(events[1], MatchEndEvent)

    def test_drain_empties_queue(self):
        q = OverwolfEventQueue()
        q.push(MatchStartEvent(timestamp=100))
        q.drain()
        assert len(q.drain()) == 0

    def test_drain_empty_queue(self):
        q = OverwolfEventQueue()
        assert q.drain() == []


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:
    @pytest.mark.parametrize("event", [
        GameStatusEvent(running=True, timestamp=100),
        GameStateUpdate(state=GameState.MATCH_IN_PROGRESS, timestamp=200),
        GameModeUpdate(code="0023", name="Control", timestamp=300),
        BattleTagUpdate(battle_tag="Player#1234", timestamp=400),
        GameTypeUpdate(game_type=GameType.RANKED, timestamp=500),
        QueueTypeUpdate(queue_type=QueueType.ROLE_QUEUE, timestamp=600),
        MapUpdate(code="212", name="King's Row", timestamp=700),
        PseudoMatchIdUpdate(pseudo_match_id="abc-123", timestamp=800),
        MatchOutcomeUpdate(outcome=MatchOutcome.VICTORY, timestamp=900),
        EliminationsUpdate(count=6, timestamp=1000),
        ObjectiveKillsUpdate(count=2, timestamp=1100),
        DeathsUpdate(count=3, timestamp=1200),
        AssistsUpdate(count=4, timestamp=1300),
        EliminationEvent(total=8, timestamp=1400),
        DeathEvent(total=3, timestamp=1500),
        AssistEvent(total=5, timestamp=1600),
        MatchStartEvent(timestamp=1700),
        MatchEndEvent(timestamp=1800),
        RoundStartEvent(timestamp=1900),
        RoundEndEvent(timestamp=2000),
        RespawnEvent(timestamp=2100),
        ReviveEvent(timestamp=2200),
    ])
    def test_roundtrip_simple(self, event):
        line = serialize_event(event, frame=42)
        frame, restored = deserialize_event(line)
        assert frame == 42
        assert type(restored) is type(event)
        assert restored == event

    def test_roundtrip_roster_update(self):
        entry = RosterEntry(
            player_name="TEST", battlenet_tag="TEST#1234",
            is_local=True, is_teammate=True,
            hero_name="ANA", hero_role="SUPPORT", team=0,
            kills=5, deaths=1, assists=3,
            damage=1234.5, healed=5678.9, mitigated=0,
        )
        event = RosterUpdate(slot=3, entry=entry, timestamp=100)
        line = serialize_event(event, frame=99)
        frame, restored = deserialize_event(line)
        assert frame == 99
        assert isinstance(restored, RosterUpdate)
        assert restored.slot == 3
        assert restored.entry.player_name == "TEST"
        assert restored.entry.damage == 1234.5

    def test_deserialize_unknown_class(self):
        with pytest.raises(ValueError, match="Unknown event class"):
            deserialize_event('{"frame": 0, "cls": "FakeEvent"}')


# ---------------------------------------------------------------------------
# JSONL recording writer / loader
# ---------------------------------------------------------------------------

class TestRecordingIO:
    def test_write_and_load(self, tmp_path):
        path = tmp_path / "test.overwolf.jsonl"
        writer = OverwolfRecordingWriter(path)
        writer.write(MatchStartEvent(timestamp=100), frame=0)
        writer.write(GameStateUpdate(state=GameState.MATCH_IN_PROGRESS, timestamp=200), frame=5)
        writer.write(MatchEndEvent(timestamp=300), frame=100)
        writer.close()

        events = load_overwolf_events(path)
        assert len(events) == 3
        assert events[0] == (0, MatchStartEvent(timestamp=100))
        assert events[1] == (5, GameStateUpdate(state=GameState.MATCH_IN_PROGRESS, timestamp=200))
        assert events[2] == (100, MatchEndEvent(timestamp=300))

    def test_load_skips_bad_lines(self, tmp_path):
        path = tmp_path / "test.overwolf.jsonl"
        path.write_text('{"frame": 0, "cls": "MatchStartEvent", "timestamp": 100}\nbad line\n')
        events = load_overwolf_events(path)
        assert len(events) == 1

    def test_load_sorted_by_frame(self, tmp_path):
        path = tmp_path / "test.overwolf.jsonl"
        # Write out of order
        lines = [
            serialize_event(MatchEndEvent(timestamp=300), frame=100),
            serialize_event(MatchStartEvent(timestamp=100), frame=0),
        ]
        path.write_text("\n".join(lines) + "\n")
        events = load_overwolf_events(path)
        assert events[0][0] == 0
        assert events[1][0] == 100

    def test_load_rebase_global_frame_offset(self, tmp_path):
        """Old recordings with global frame numbers get rebased to start at 0."""
        path = tmp_path / "test.overwolf.jsonl"
        lines = [
            serialize_event(MatchStartEvent(timestamp=100), frame=5000),
            serialize_event(MatchEndEvent(timestamp=300), frame=5200),
        ]
        path.write_text("\n".join(lines) + "\n")
        events = load_overwolf_events(path)
        assert events[0][0] == 0
        assert events[1][0] == 200

    def test_writer_frame_offset(self, tmp_path):
        """Writer subtracts frame_offset so events are recording-relative."""
        path = tmp_path / "test.overwolf.jsonl"
        writer = OverwolfRecordingWriter(path, frame_offset=1000)
        writer.write(MatchStartEvent(timestamp=100), 1000)
        writer.write(MatchEndEvent(timestamp=200), 1050)
        writer.close()
        events = load_overwolf_events(path)
        assert events[0][0] == 0
        assert events[1][0] == 50


class TestMapCodes:
    def test_runasapi(self):
        from overwatchlooker.overwolf import MAP_CODES
        assert MAP_CODES["3762"] == "Runasapi"

    def test_push_mode(self):
        from overwatchlooker.overwolf import MODE_CODES
        assert MODE_CODES["0064"] == "Push"

    def test_clash_mode(self):
        from overwatchlooker.overwolf import MODE_CODES
        assert MODE_CODES["0077"] == "Clash"
