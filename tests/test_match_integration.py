"""Integration test: simulate a full attack-side King's Row defeat with all systems.

Scenario
--------
Competitive Role Queue on King's Row (Hybrid), attack side.
We capture point A but fail to push the payload past the first checkpoint.
Two rounds: attack (we capture obj), then defense cut short (they full-hold us
on payload). Result: DEFEAT.

Timeline (epoch-ms offsets from match start at t=0):
  t=0        MatchStart, RoundStart (attack)
  t=5000     Initial roster (10 players, 5v5)
  t=30000    Subtitle OCR: YOURNAME switched to Reinhardt (confirms Overwolf)
  t=60000    Chat OCR: YOURNAME joined the voice channel
  t=90000    RAGEQUIT leaves mid-match (chat OCR)
  t=95000    Overwolf roster: RAGEQUIT disappears, slot goes empty
  t=100000   REPLACEMENT joins (chat OCR)
  t=105000   Overwolf roster: REPLACEMENT appears in RAGEQUIT's slot
  t=120000   Tab screenshot captured
  t=150000   YOURNAME swaps Reinhardt -> Winston (Overwolf roster update)
  t=155000   Subtitle OCR also sees "YOURNAME switched to Winston"
  t=180000   RoundEnd (attack round over — point captured)
  t=185000   RoundStart (defense / payload phase)
  t=300000   RoundEnd (they full-hold us)
  t=305000   MatchOutcomeUpdate: DEFEAT
  t=310000   MatchEnd

Assertions verify the final snapshot has all the right data from all sources.
"""

import pytest

from overwatchlooker.match_state import (
    HeroSource,
    MatchResult,
    MatchState,
    PlayerRole,
    ResultSource,
    TeamSide,
    format_match_state,
)
from overwatchlooker.overwolf import (
    GameModeUpdate,
    GameType,
    GameTypeUpdate,
    MapUpdate,
    MatchOutcome,
    MatchOutcomeUpdate,
    MatchStartEvent,
    PseudoMatchIdUpdate,
    QueueType,
    QueueTypeUpdate,
    RosterEntry,
    RosterUpdate,
    RoundEndEvent,
    RoundStartEvent,
)
from overwatchlooker.tray import App


# ── Helpers ──────────────────────────────────────────────────────────────

BASE_TS = 1_700_000_000_000  # arbitrary epoch-ms anchor


def _ts(offset_ms: int) -> int:
    return BASE_TS + offset_ms


def _roster(slot, name, tag, *, local=False, teammate=True, hero="",
            role="", team=0, k=0, d=0, a=0, dmg=0, heal=0, mit=0, ts=0):
    return RosterUpdate(
        slot=slot,
        entry=RosterEntry(
            player_name=name, battlenet_tag=tag,
            is_local=local, is_teammate=teammate,
            hero_name=hero, hero_role=role, team=team,
            kills=k, deaths=d, assists=a,
            damage=dmg, healed=heal, mitigated=mit,
        ),
        timestamp=_ts(ts),
    )


# ── Fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def scenario():
    """Run the full scenario and return the captured snapshot."""
    app = App()
    captured: list[MatchState] = []

    # Patch _trigger_match_end to capture the snapshot synchronously
    def _capture_trigger():
        ms = app._match_state
        if ms._analysis_triggered:
            return
        snapshot = ms.snapshot()
        captured.append(snapshot)
        # Still reset like the real method
        if app._detector is not None:
            app._detector.reset_match()
        if app._chat_system is not None:
            app._chat_system.reset_match()
        app._match_state = MatchState()

    app._trigger_match_end = _capture_trigger

    # ── Pre-match info (arrives before MatchStart) ──
    app._on_overwolf_event(MapUpdate(code="212", name="King's Row", timestamp=_ts(-2000)))
    app._on_overwolf_event(GameModeUpdate(code="0022", name="Hybrid", timestamp=_ts(-2000)))
    app._on_overwolf_event(GameTypeUpdate(game_type=GameType.RANKED, timestamp=_ts(-1500)))
    app._on_overwolf_event(QueueTypeUpdate(queue_type=QueueType.ROLE_QUEUE, timestamp=_ts(-1500)))
    app._on_overwolf_event(PseudoMatchIdUpdate(pseudo_match_id="match-abc-123", timestamp=_ts(-500)))

    # ── Match starts ──
    app._on_overwolf_event(MatchStartEvent(timestamp=_ts(0)))
    app._on_overwolf_event(RoundStartEvent(timestamp=_ts(0)))

    # ── Initial roster (t=5000) ──
    # Allies (team 0)
    app._on_overwolf_event(_roster(0, "YourName", "YourName#1234",
                                    local=True, hero="Reinhardt", role="TANK",
                                    team=0, k=0, d=0, a=0, dmg=0, mit=0, ts=5000))
    app._on_overwolf_event(_roster(1, "DpsPlayer1", "DpsPlayer1#2222",
                                    hero="Tracer", role="DAMAGE",
                                    team=0, ts=5000))
    app._on_overwolf_event(_roster(2, "DpsPlayer2", "DpsPlayer2#3333",
                                    hero="Sojourn", role="DAMAGE",
                                    team=0, ts=5000))
    app._on_overwolf_event(_roster(3, "Healer1", "Healer1#4444",
                                    hero="Ana", role="SUPPORT",
                                    team=0, ts=5000))
    app._on_overwolf_event(_roster(4, "RageQuit", "RageQuit#5555",
                                    hero="Lucio", role="SUPPORT",
                                    team=0, ts=5000))

    # Enemies (team 1) — hero/role often empty for enemies
    app._on_overwolf_event(_roster(5, "Enemy1", "Enemy1#6666",
                                    teammate=False, team=1, ts=5000))
    app._on_overwolf_event(_roster(6, "Enemy2", "Enemy2#7777",
                                    teammate=False, team=1, ts=5000))
    app._on_overwolf_event(_roster(7, "Enemy3", "Enemy3#8888",
                                    teammate=False, team=1, ts=5000))
    app._on_overwolf_event(_roster(8, "Enemy4", "Enemy4#9999",
                                    teammate=False, team=1, ts=5000))
    app._on_overwolf_event(_roster(9, "Enemy5", "Enemy5#0000",
                                    teammate=False, team=1, ts=5000))

    # ── Subtitle OCR confirms hero (t=30s) ──
    app._on_hero_switch("YOURNAME", "Reinhardt", 30.0)

    # ── Chat OCR: voice channel join (t=60s) ──
    app._on_player_change("YOURNAME", "joined", 60.0)

    # ── Tab screenshot (t=120s) ──
    app.store_valid_tab(b"fake_png_tab_1", 120.0, "tab_120s.png")

    # ── RageQuit leaves (t=90s chat OCR) ──
    app._on_player_change("RAGEQUIT", "left", 90.0)

    # ── Overwolf: updated roster without RageQuit's hero (t=95s) ──
    # (Overwolf typically sends an update with empty hero for the leaver)
    app._on_overwolf_event(_roster(4, "RageQuit", "RageQuit#5555",
                                    hero="", role="",
                                    team=0, k=1, d=4, a=2, dmg=800, heal=3200, ts=95000))

    # ── Replacement joins (t=100s chat OCR) ──
    app._on_player_change("REPLACEMENT", "joined", 100.0)

    # ── Overwolf: Replacement appears in slot 4 (t=105s) ──
    app._on_overwolf_event(_roster(4, "Replacement", "Replacement#1111",
                                    hero="Kiriko", role="SUPPORT",
                                    team=0, k=0, d=0, a=0, dmg=0, heal=0, ts=105000))

    # ── Hero swap: YourName Reinhardt -> Winston (t=150s via Overwolf) ──
    app._on_overwolf_event(_roster(0, "YourName", "YourName#1234",
                                    local=True, hero="Winston", role="TANK",
                                    team=0, k=8, d=3, a=5, dmg=6000, mit=4500, ts=150000))

    # ── Subtitle OCR also picks up the swap (t=155s) — should be deduped ──
    app._on_hero_switch("YOURNAME", "Winston", 155.0)

    # ── Round 1 ends (attack, point captured) ──
    app._on_overwolf_event(RoundEndEvent(timestamp=_ts(180000)))

    # ── Round 2 starts (payload phase) ──
    app._on_overwolf_event(RoundStartEvent(timestamp=_ts(185000)))

    # ── Mid-match roster update with updated stats (t=250s) ──
    app._on_overwolf_event(_roster(0, "YourName", "YourName#1234",
                                    local=True, hero="Winston", role="TANK",
                                    team=0, k=12, d=5, a=8, dmg=9500, mit=7200, ts=250000))
    app._on_overwolf_event(_roster(1, "DpsPlayer1", "DpsPlayer1#2222",
                                    hero="Tracer", role="DAMAGE",
                                    team=0, k=15, d=4, a=3, dmg=11000, ts=250000))
    app._on_overwolf_event(_roster(5, "Enemy1", "Enemy1#6666",
                                    teammate=False, team=1,
                                    k=10, d=7, a=6, dmg=8000, ts=250000))

    # ── Round 2 ends (full hold, payload stuck) ──
    app._on_overwolf_event(RoundEndEvent(timestamp=_ts(300000)))

    # ── Match outcome + end ──
    app._on_overwolf_event(MatchOutcomeUpdate(
        outcome=MatchOutcome.DEFEAT, timestamp=_ts(305000)))

    assert len(captured) == 1, f"Expected exactly 1 snapshot, got {len(captured)}"
    return captured[0]


# ── Tests ────────────────────────────────────────────────────────────────

class TestMatchMetadata:
    def test_map(self, scenario):
        assert scenario.map_name == "King's Row"
        assert scenario.map_code == "212"

    def test_mode(self, scenario):
        assert scenario.mode == "Hybrid"
        assert scenario.mode_code == "0022"

    def test_game_type(self, scenario):
        assert scenario.game_type == GameType.RANKED

    def test_queue_type(self, scenario):
        assert scenario.queue_type == QueueType.ROLE_QUEUE

    def test_result(self, scenario):
        assert scenario.result == MatchResult.DEFEAT
        assert scenario.result_source == ResultSource.OVERWOLF

    def test_pseudo_match_id(self, scenario):
        assert scenario.pseudo_match_id == "match-abc-123"

    def test_started_at(self, scenario):
        assert scenario.started_at == _ts(0)

    def test_duration(self, scenario):
        # Duration is from first RoundStart (t=0) to MatchOutcomeUpdate (t=305000)
        assert scenario.ended_at == _ts(305000)
        assert scenario.duration == 305000


class TestRounds:
    def test_round_count(self, scenario):
        assert len(scenario.rounds) == 2

    def test_round_1_attack(self, scenario):
        r1 = scenario.rounds[0]
        assert r1.started_at == _ts(0)
        assert r1.ended_at == _ts(180000)
        # 3 minutes attack round
        assert r1.ended_at - r1.started_at == 180000

    def test_round_2_payload(self, scenario):
        r2 = scenario.rounds[1]
        assert r2.started_at == _ts(185000)
        assert r2.ended_at == _ts(300000)
        assert r2.ended_at - r2.started_at == 115000


class TestTeams:
    def test_local_team(self, scenario):
        assert scenario._local_team == 0

    def test_ally_count(self, scenario):
        allies = [p for p in scenario.players.values()
                  if p.team_side == TeamSide.ALLY]
        # YourName, DpsPlayer1, DpsPlayer2, Healer1, RageQuit, Replacement = 6
        # (RageQuit left but still in roster, Replacement joined)
        assert len(allies) == 6

    def test_enemy_count(self, scenario):
        enemies = [p for p in scenario.players.values()
                   if p.team_side == TeamSide.ENEMY]
        assert len(enemies) == 5

    def test_all_allies_have_team_side(self, scenario):
        for tag in ["YOURNAME#1234", "DPSPLAYER1#2222", "DPSPLAYER2#3333",
                     "HEALER1#4444", "RAGEQUIT#5555", "REPLACEMENT#1111"]:
            assert scenario.players[tag].team_side == TeamSide.ALLY

    def test_all_enemies_have_team_side(self, scenario):
        tags = {"ENEMY1#6666": 1, "ENEMY2#7777": 2, "ENEMY3#8888": 3,
                "ENEMY4#9999": 4, "ENEMY5#0000": 5}
        for tag in tags:
            assert scenario.players[tag].team_side == TeamSide.ENEMY


class TestLocalPlayer:
    def test_local_player_identified(self, scenario):
        local = scenario.local_player
        assert local is not None
        assert local.player_name == "YOURNAME"
        assert local.battletag == "YourName#1234"
        assert local.is_local is True

    def test_local_player_role(self, scenario):
        assert scenario.local_player.role == PlayerRole.TANK

    def test_local_player_final_stats(self, scenario):
        s = scenario.local_player.stats
        assert s.kills == 12
        assert s.deaths == 5
        assert s.assists == 8
        assert s.damage == 9500
        assert s.mitigation == 7200


class TestHeroSwaps:
    def test_local_player_hero_history(self, scenario):
        swaps = scenario.local_player.hero_swaps
        assert len(swaps) == 2
        assert swaps[0].hero == "Reinhardt"
        assert swaps[1].hero == "Winston"

    def test_first_hero_from_overwolf(self, scenario):
        swap = scenario.local_player.hero_swaps[0]
        assert swap.source == HeroSource.OVERWOLF_ROSTER
        assert swap.detected_at == _ts(5000)

    def test_hero_swap_from_overwolf(self, scenario):
        swap = scenario.local_player.hero_swaps[1]
        assert swap.source == HeroSource.OVERWOLF_ROSTER
        assert swap.detected_at == _ts(150000)

    def test_hero_swap_stats_snapshot(self, scenario):
        """Stats at the moment of the swap should be captured."""
        swap = scenario.local_player.hero_swaps[1]
        assert swap.stats_at_detection is not None
        assert swap.stats_at_detection.kills == 8
        assert swap.stats_at_detection.damage == 6000

    def test_subtitle_swap_deduped(self, scenario):
        """Subtitle OCR at t=155s should NOT create a third swap (Winston already set)."""
        assert len(scenario.local_player.hero_swaps) == 2

    def test_subtitle_initial_deduped(self, scenario):
        """Subtitle OCR at t=30s for Reinhardt should be deduped with Overwolf's."""
        assert len(scenario.local_player.hero_swaps) == 2

    def test_current_hero(self, scenario):
        assert scenario.local_player.current_hero == "Winston"

    def test_dps_hero_unchanged(self, scenario):
        dps = scenario.players["DPSPLAYER1#2222"]
        assert len(dps.hero_swaps) == 1
        assert dps.current_hero == "Tracer"


class TestPlayerLeaveAndReplace:
    def test_ragequit_left_at(self, scenario):
        rq = scenario.players["RAGEQUIT#5555"]
        assert rq.left_at == 90000  # from chat OCR at sim_time=90.0

    def test_ragequit_last_stats(self, scenario):
        rq = scenario.players["RAGEQUIT#5555"]
        assert rq.stats.kills == 1
        assert rq.stats.deaths == 4
        assert rq.stats.healing == 3200

    def test_ragequit_hero_was_lucio(self, scenario):
        rq = scenario.players["RAGEQUIT#5555"]
        assert rq.hero_swaps[0].hero == "Lucio"

    def test_replacement_joined_at(self, scenario):
        rep = scenario.players["REPLACEMENT#1111"]
        assert rep.joined_at == _ts(105000)  # from first roster update

    def test_replacement_hero(self, scenario):
        rep = scenario.players["REPLACEMENT#1111"]
        assert rep.current_hero == "Kiriko"
        assert rep.role == PlayerRole.SUPPORT

    def test_replacement_is_ally(self, scenario):
        rep = scenario.players["REPLACEMENT#1111"]
        assert rep.team_side == TeamSide.ALLY
        assert rep.team == 0


class TestTabScreenshots:
    def test_latest_tab_stored(self, scenario):
        assert scenario.latest_tab is not None
        assert scenario.latest_tab.filename == "tab_120s.png"
        assert scenario.latest_tab.sim_time == 120.0


class TestEnemyPlayers:
    def test_enemy_no_hero_info(self, scenario):
        """Enemies typically don't have hero info from Overwolf."""
        e = scenario.players["ENEMY2#7777"]
        assert len(e.hero_swaps) == 0
        assert e.current_hero is None

    def test_enemy_with_updated_stats(self, scenario):
        """Enemy1 got a stats update at t=250s."""
        e = scenario.players["ENEMY1#6666"]
        assert e.stats.kills == 10
        assert e.stats.damage == 8000


class TestFormatOutput:
    def test_format_contains_key_info(self, scenario):
        output = format_match_state(scenario)
        assert "King's Row" in output
        assert "DEFEAT" in output
        assert "Hybrid" in output
        assert "Competitive" in output
        assert "Role Queue" in output

    def test_format_contains_players(self, scenario):
        output = format_match_state(scenario)
        assert "YourName#1234" in output
        assert "DpsPlayer1#2222" in output
        assert "Enemy1#6666" in output

    def test_format_contains_hero_swap_arrow(self, scenario):
        output = format_match_state(scenario)
        assert "Reinhardt" in output
        assert "Winston" in output

    def test_format_contains_rounds(self, scenario):
        output = format_match_state(scenario)
        assert "Rounds: 2" in output

    def test_format_contains_stats(self, scenario):
        output = format_match_state(scenario)
        assert "12/5/8" in output
        assert "9500 dmg" in output
