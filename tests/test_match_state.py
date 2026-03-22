"""Tests for match_state module: dataclasses, player management, formatting."""

from overwatchlooker.match_state import (
    HeroPanel,
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
    build_mcp_payload,
    format_match_state,
)
from overwatchlooker.overwolf import GameType, QueueType


class TestMatchStateCreation:
    def test_empty_state(self):
        ms = MatchState()
        assert ms.players == {}
        assert ms.hero_tabs == {}
        assert ms.latest_tab is None
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

    def test_format_contains_rank(self):
        ms = self._make_match()
        ms.rank_min = "Bronze 2"
        ms.rank_max = "Gold 1"
        ms.is_wide_match = True
        output = format_match_state(ms)
        assert "Bronze 2" in output
        assert "Gold 1" in output
        assert "WIDE" in output

    def test_format_contains_bans(self):
        ms = self._make_match()
        ms.hero_bans = ["Mercy", "Zarya"]
        output = format_match_state(ms)
        assert "Bans: Mercy, Zarya" in output

    def test_format_contains_hero_panel_stats(self):
        from overwatchlooker.match_state import HeroPanel
        ms = self._make_match()
        local = ms.players["LUNAVOD"]
        local.is_local = True
        local.hero_panels = [HeroPanel(
            hero_name="Reinhardt", crop_png=b"",
            ocr_stats=[
                {"label": "CHARGE KILLS", "value": "3", "is_featured": False},
                {"label": "PLAYERS SAVED", "value": "5", "is_featured": True},
            ],
        )]
        output = format_match_state(ms)
        assert "Reinhardt Stats" in output
        assert "CHARGE KILLS: 3" in output
        assert "PLAYERS SAVED: 5" in output


class TestBuildMcpPayload:
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
                RoundInfo(started_at=102000, ended_at=486000),
            ],
        )
        ms._local_team = 0

        tank = ms.get_or_create_player("LUNAVOD")
        tank.battletag = "LUNAVOD#1234"
        tank.team = 0
        tank.team_side = TeamSide.ALLY
        tank.role = PlayerRole.TANK
        tank.is_local = True
        tank.hero_swaps = [
            HeroSwap(hero="Reinhardt", detected_at=0, source=HeroSource.OVERWOLF_ROSTER),
            HeroSwap(hero="Winston", detected_at=150000, source=HeroSource.OVERWOLF_ROSTER),
        ]
        tank.stats = StatsSnapshot(kills=15, deaths=3, assists=8,
                                   damage=12500, healing=0, mitigation=8900)

        enemy = ms.get_or_create_player("ENEMY1")
        enemy.battletag = "ENEMY1#9999"
        enemy.team = 1
        enemy.team_side = TeamSide.ENEMY
        enemy.hero_swaps = [
            HeroSwap(hero="Roadhog", detected_at=0, source=HeroSource.OVERWOLF_ROSTER),
        ]
        enemy.stats = StatsSnapshot(kills=4, deaths=6, assists=2,
                                    damage=3000, healing=0, mitigation=0)

        return ms

    def test_basic_fields(self):
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        assert payload["map_name"] == "King's Row"
        assert payload["mode"] == "HYBRID"
        assert payload["queue_type"] == "COMPETITIVE"
        assert payload["result"] == "VICTORY"
        assert payload["duration"] == "8:06"

    def test_quickplay_queue_type(self):
        ms = self._make_match()
        ms.game_type = GameType.UNRANKED
        payload = build_mcp_payload(ms)
        assert payload["queue_type"] == "QUICKPLAY"

    def test_players_array(self):
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        assert len(payload["players"]) == 2
        names = {p["player_name"] for p in payload["players"]}
        assert "LUNAVOD#1234" in names
        assert "ENEMY1#9999" in names

    def test_player_role_dps(self):
        ms = self._make_match()
        dps = ms.get_or_create_player("DPS1")
        dps.battletag = "DPS1#1111"
        dps.team = 0
        dps.team_side = TeamSide.ALLY
        dps.role = PlayerRole.DAMAGE
        payload = build_mcp_payload(ms)
        dps_player = next(p for p in payload["players"] if p["player_name"] == "DPS1#1111")
        assert dps_player["role"] == "DPS"

    def test_player_stats(self):
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        tank = next(p for p in payload["players"] if p["player_name"] == "LUNAVOD#1234")
        assert tank["eliminations"] == 15
        assert tank["deaths"] == 3
        assert tank["damage"] == 12500

    def test_heroes_array(self):
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        tank = next(p for p in payload["players"] if p["player_name"] == "LUNAVOD#1234")
        assert len(tank["heroes"]) == 2
        assert tank["heroes"][0]["hero_name"] == "Reinhardt"
        assert tank["heroes"][0]["started_at"] == [0]
        assert tank["heroes"][1]["hero_name"] == "Winston"
        assert tank["heroes"][1]["started_at"] == [150]

    def test_hero_panel_stats_attached(self):
        ms = self._make_match()
        local = ms.players["LUNAVOD"]
        local.hero_panels = [HeroPanel(
            hero_name="Reinhardt", crop_png=b"",
            ocr_stats=[
                {"label": "CHARGE KILLS", "value": "3", "is_featured": False},
            ],
        )]
        payload = build_mcp_payload(ms)
        tank = next(p for p in payload["players"] if p["player_name"] == "LUNAVOD#1234")
        rein = tank["heroes"][0]
        assert rein["hero_name"] == "Reinhardt"
        assert rein["stats"] == [{"label": "CHARGE KILLS", "value": "3", "is_featured": False}]

    def test_rank_fields(self):
        ms = self._make_match()
        ms.rank_min = "Bronze 2"
        ms.rank_max = "Gold 1"
        ms.is_wide_match = True
        payload = build_mcp_payload(ms)
        assert payload["rank_min"] == "Bronze 2"
        assert payload["rank_max"] == "Gold 1"
        assert payload["is_wide_match"] is True

    def test_no_rank_omitted(self):
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        assert "rank_min" not in payload
        assert "rank_max" not in payload

    def test_is_self_flag(self):
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        tank = next(p for p in payload["players"] if p["player_name"] == "LUNAVOD#1234")
        assert tank["is_self"] is True
        enemy = next(p for p in payload["players"] if p["player_name"] == "ENEMY1#9999")
        assert enemy["is_self"] is False

    def test_disconnected_players_excluded(self):
        """Players whose slot was taken by a replacement are excluded."""
        ms = self._make_match()
        # Disconnected support on slot 5 — left mid-match with some stats
        dc = ms.get_or_create_player("DISCONNECTED")
        dc.battletag = "Disconnected#1111"
        dc.team = 0
        dc.team_side = TeamSide.ALLY
        dc.role = PlayerRole.SUPPORT
        dc.slot = 5
        dc.stats = StatsSnapshot(kills=2, deaths=3, assists=5,
                                 damage=500, healing=1200, mitigation=0)
        dc.hero_swaps = [
            HeroSwap(hero="Mercy", detected_at=1000, source=HeroSource.OVERWOLF_ROSTER),
        ]
        # Replacement takes the same slot later
        repl = ms.get_or_create_player("REPLACEMENT")
        repl.battletag = "Replacement#2222"
        repl.team = 0
        repl.team_side = TeamSide.ALLY
        repl.role = PlayerRole.SUPPORT
        repl.slot = 5
        repl.stats = StatsSnapshot(kills=1, deaths=0, assists=3,
                                   damage=200, healing=800, mitigation=0)
        repl.hero_swaps = [
            HeroSwap(hero="Ana", detected_at=200000, source=HeroSource.OVERWOLF_ROSTER),
        ]
        payload = build_mcp_payload(ms)
        names = {p["player_name"] for p in payload["players"]}
        assert "Disconnected#1111" not in names
        assert "Replacement#2222" in names
        # Original players still present
        assert "LUNAVOD#1234" in names
        assert "ENEMY1#9999" in names

    def test_unique_slots_not_excluded(self):
        """Players with unique slots are never excluded."""
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        names = {p["player_name"] for p in payload["players"]}
        assert "LUNAVOD#1234" in names
        assert "ENEMY1#9999" in names

    def test_swap_snapshots(self):
        """Cumulative stats at each hero swap are included."""
        ms = self._make_match()
        # Give the tank's swaps stats_at_detection
        ms.players["LUNAVOD"].hero_swaps[0].stats_at_detection = StatsSnapshot(
            kills=0, deaths=0, assists=0, damage=0, healing=0, mitigation=0)
        ms.players["LUNAVOD"].hero_swaps[1].stats_at_detection = StatsSnapshot(
            kills=8, deaths=2, assists=4, damage=6000, healing=0, mitigation=4000)
        payload = build_mcp_payload(ms)
        tank = next(p for p in payload["players"] if p["player_name"] == "LUNAVOD#1234")
        assert "swap_snapshots" in tank
        assert len(tank["swap_snapshots"]) == 2
        snap = tank["swap_snapshots"][1]
        assert snap["eliminations"] == 8
        assert snap["damage"] == 6000
        assert snap["time"] == 150

    def test_swap_snapshots_omitted_when_none(self):
        """swap_snapshots is not included when stats_at_detection is None."""
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        enemy = next(p for p in payload["players"] if p["player_name"] == "ENEMY1#9999")
        assert "swap_snapshots" not in enemy

    def test_banned_heroes(self):
        ms = self._make_match()
        ms.hero_bans = ["Mercy", "Zarya"]
        payload = build_mcp_payload(ms)
        assert payload["banned_heroes"] == ["Mercy", "Zarya"]

    def test_banned_heroes_omitted_when_empty(self):
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        assert "banned_heroes" not in payload

    def test_joined_at(self):
        ms = self._make_match()
        # Player joined 30s into the match
        ms.players["ENEMY1"].joined_at = 30000
        payload = build_mcp_payload(ms)
        enemy = next(p for p in payload["players"] if p["player_name"] == "ENEMY1#9999")
        assert enemy["joined_at"] == 30

    def test_in_party_flag(self):
        ms = self._make_match()
        ms.players["LUNAVOD"].in_party = True
        payload = build_mcp_payload(ms)
        tank = next(p for p in payload["players"] if p["player_name"] == "LUNAVOD#1234")
        assert tank["in_party"] is True
        enemy = next(p for p in payload["players"] if p["player_name"] == "ENEMY1#9999")
        assert enemy["in_party"] is False

    def test_format_contains_party_marker(self):
        ms = self._make_match()
        ms.players["LUNAVOD"].in_party = True
        output = format_match_state(ms)
        assert "[P]" in output

    def test_backfill_in_payload(self):
        ms = self._make_match()
        ms.is_backfill = True
        payload = build_mcp_payload(ms)
        assert payload["is_backfill"] is True

    def test_backfill_default_false(self):
        ms = self._make_match()
        payload = build_mcp_payload(ms)
        assert payload["is_backfill"] is False

    def test_format_contains_backfill(self):
        ms = self._make_match()
        ms.is_backfill = True
        output = format_match_state(ms)
        assert "(BACKFILL)" in output
