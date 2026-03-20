"""Tests for match_state module: dataclasses, player management, formatting."""

from overwatchlooker.match_state import (
    HeroSource,
    HeroSwap,
    MatchResult,
    MatchState,
    PlayerRole,
    PlayerState,
    ResultSource,
    RoundInfo,
    StatsSnapshot,
    TeamSide,
    format_match_state,
)
from overwatchlooker.overwolf import GameType, QueueType


class TestMatchStateCreation:
    def test_empty_state(self):
        ms = MatchState()
        assert ms.players == {}
        assert ms.tab_screenshots == []
        assert ms.rounds == []
        assert ms.result is None
        assert ms.started_at is None
        assert ms.ended_at is None
        assert ms._local_team is None

    def test_get_or_create_player_new(self):
        ms = MatchState()
        p = ms.get_or_create_player("TestPlayer")
        assert p.player_name == "TESTPLAYER"
        assert "TESTPLAYER" in ms.players

    def test_get_or_create_player_existing(self):
        ms = MatchState()
        p1 = ms.get_or_create_player("TestPlayer")
        p1.battletag = "Test#1234"
        p2 = ms.get_or_create_player("testplayer")
        assert p2 is p1
        assert p2.battletag == "Test#1234"

    def test_local_player_none(self):
        ms = MatchState()
        assert ms.local_player is None

    def test_local_player_found(self):
        ms = MatchState()
        p = ms.get_or_create_player("Me")
        p.is_local = True
        assert ms.local_player is p

    def test_duration_none(self):
        ms = MatchState()
        assert ms.duration is None

    def test_duration_computed(self):
        ms = MatchState(started_at=1000, ended_at=5000)
        assert ms.duration == 4000

    def test_duration_partial(self):
        ms = MatchState(started_at=1000)
        assert ms.duration is None

    def test_duration_from_round_start(self):
        """Duration uses first round start, not match start (excludes hero select)."""
        ms = MatchState(started_at=0, ended_at=10000,
                        rounds=[RoundInfo(started_at=2000, ended_at=10000)])
        assert ms.duration == 8000  # from round start, not match start


class TestPlayerState:
    def test_current_hero_none(self):
        p = PlayerState(player_name="TEST")
        assert p.current_hero is None

    def test_current_hero_last_swap(self):
        p = PlayerState(player_name="TEST", hero_swaps=[
            HeroSwap(hero="Reinhardt", detected_at=1000, source=HeroSource.OVERWOLF_ROSTER),
            HeroSwap(hero="Winston", detected_at=2000, source=HeroSource.OVERWOLF_ROSTER),
        ])
        assert p.current_hero == "Winston"


class TestTeamSideResolution:
    def test_resolve_team_sides(self):
        ms = MatchState()
        ms._local_team = 0
        p_ally = ms.get_or_create_player("Ally")
        p_ally.team = 0
        p_enemy = ms.get_or_create_player("Enemy")
        p_enemy.team = 1
        ms.resolve_team_sides()
        assert p_ally.team_side == TeamSide.ALLY
        assert p_enemy.team_side == TeamSide.ENEMY

    def test_resolve_no_local_team(self):
        ms = MatchState()
        p = ms.get_or_create_player("Player")
        p.team = 0
        ms.resolve_team_sides()
        assert p.team_side is None


class TestSnapshot:
    def test_snapshot_deep_copies(self):
        ms = MatchState(map_name="King's Row")
        p = ms.get_or_create_player("Player")
        p.hero_swaps.append(HeroSwap(hero="Ana", detected_at=1000,
                                      source=HeroSource.SUBTITLE_OCR))
        snap = ms.snapshot()
        assert snap.map_name == "King's Row"
        assert len(snap.players["PLAYER"].hero_swaps) == 1
        # Modifying original doesn't affect snapshot
        ms.map_name = "Dorado"
        assert snap.map_name == "King's Row"

    def test_snapshot_sets_triggered(self):
        ms = MatchState()
        assert ms._analysis_triggered is False
        ms.snapshot()
        assert ms._analysis_triggered is True


class TestFormatMatchState:
    def _make_match(self) -> MatchState:
        ms = MatchState(
            map_name="King's Row",
            mode="Hybrid",
            game_type=GameType.RANKED,
            queue_type=QueueType.ROLE_QUEUE,
            result=MatchResult.VICTORY,
            result_source=ResultSource.OVERWOLF,
            started_at=0,
            ended_at=486000,
            rounds=[
                RoundInfo(started_at=0, ended_at=102000),
                RoundInfo(started_at=102000, ended_at=237000),
                RoundInfo(started_at=237000, ended_at=486000),
            ],
        )
        ms._local_team = 0

        # Allies
        tank = ms.get_or_create_player("LUNAVOD")
        tank.battletag = "LUNAVOD#1234"
        tank.team = 0
        tank.team_side = TeamSide.ALLY
        tank.role = PlayerRole.TANK
        tank.hero_swaps = [
            HeroSwap(hero="Reinhardt", detected_at=0, source=HeroSource.OVERWOLF_ROSTER),
            HeroSwap(hero="Winston", detected_at=150000, source=HeroSource.OVERWOLF_ROSTER),
        ]
        tank.stats = StatsSnapshot(kills=15, deaths=3, assists=8,
                                   damage=12500, healing=0, mitigation=8900)

        dps = ms.get_or_create_player("PLAYER2")
        dps.battletag = "PLAYER2#5678"
        dps.team = 0
        dps.team_side = TeamSide.ALLY
        dps.role = PlayerRole.DAMAGE
        dps.hero_swaps = [
            HeroSwap(hero="Tracer", detected_at=0, source=HeroSource.OVERWOLF_ROSTER),
        ]
        dps.stats = StatsSnapshot(kills=8, deaths=5, assists=3,
                                  damage=6200, healing=0, mitigation=0)

        # Enemy
        enemy = ms.get_or_create_player("ENEMY1")
        enemy.battletag = "ENEMY1#9999"
        enemy.team = 1
        enemy.team_side = TeamSide.ENEMY
        enemy.stats = StatsSnapshot(kills=4, deaths=6, assists=2,
                                    damage=3000, healing=0, mitigation=0)

        return ms

    def test_format_contains_map(self):
        ms = self._make_match()
        output = format_match_state(ms)
        assert "King's Row" in output

    def test_format_contains_result(self):
        ms = self._make_match()
        output = format_match_state(ms)
        assert "VICTORY" in output

    def test_format_contains_duration(self):
        ms = self._make_match()
        output = format_match_state(ms)
        assert "8:06" in output

    def test_format_contains_rounds(self):
        ms = self._make_match()
        output = format_match_state(ms)
        assert "Rounds: 3" in output

    def test_format_contains_players(self):
        ms = self._make_match()
        output = format_match_state(ms)
        assert "LUNAVOD#1234" in output
        assert "PLAYER2#5678" in output
        assert "ENEMY1#9999" in output

    def test_format_contains_hero_swap(self):
        ms = self._make_match()
        output = format_match_state(ms)
        assert "Reinhardt" in output
        assert "Winston" in output

    def test_format_contains_teams(self):
        ms = self._make_match()
        output = format_match_state(ms)
        assert "ALLY" in output
        assert "ENEMY" in output

    def test_format_empty_match(self):
        ms = MatchState()
        output = format_match_state(ms)
        assert "MATCH COMPLETE" in output
        assert "Unknown Map" in output

    def test_format_contains_stats(self):
        ms = self._make_match()
        output = format_match_state(ms)
        assert "15/3/8" in output
        assert "12500 dmg" in output
