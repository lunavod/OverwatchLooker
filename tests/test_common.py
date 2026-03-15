"""Tests for analyzers.common: schema, merge_heroes, format_match, log_cost."""

import json


from overwatchlooker.analyzers.common import (
    MATCH_SCHEMA,
    format_match,
    get_ranks_reference,
    log_cost,
    make_schema_with_extra_heroes,
    merge_heroes,
)


class TestMatchSchema:
    def test_has_required_keys(self):
        props = MATCH_SCHEMA["schema"]["properties"]
        assert "not_ow2_tab" in props
        assert "map_name" in props
        assert "players" in props
        assert "result" in props

    def test_player_has_required_fields(self):
        player_props = MATCH_SCHEMA["schema"]["properties"]["players"]["items"]["properties"]
        for field in ["team", "role", "player_name", "eliminations", "deaths",
                      "damage", "healing", "mitigation", "is_self", "hero"]:
            assert field in player_props, f"Missing player field: {field}"


class TestMakeSchemaWithExtraHeroes:
    def test_adds_extra_hero_stats(self):
        schema = make_schema_with_extra_heroes()
        assert "extra_hero_stats" in schema["schema"]["properties"]
        assert "extra_hero_stats" in schema["schema"]["required"]

    def test_does_not_mutate_original(self):
        original_keys = set(MATCH_SCHEMA["schema"]["properties"].keys())
        make_schema_with_extra_heroes()
        assert "extra_hero_stats" not in MATCH_SCHEMA["schema"]["properties"]
        assert set(MATCH_SCHEMA["schema"]["properties"].keys()) == original_keys


class TestMergeHeroes:
    def _make_data(self, hero=None, extra_hero_stats=None):
        """Minimal match dict with one self-player."""
        data = {
            "players": [
                {
                    "player_name": "PLAYER1",
                    "is_self": True,
                    "hero": hero,
                },
                {
                    "player_name": "PLAYER2",
                    "is_self": False,
                    "hero": None,
                },
            ],
        }
        if extra_hero_stats is not None:
            data["extra_hero_stats"] = extra_hero_stats
        return data

    def test_no_history_empty_heroes(self):
        data = self._make_data()
        merge_heroes(data)
        for p in data["players"]:
            assert "heroes" in p
            assert p["heroes"] == []
            assert "hero" not in p

    def test_single_hero_from_map(self):
        data = self._make_data()
        merge_heroes(data, hero_map={"PLAYER1": "Reinhardt"})
        p = data["players"][0]
        assert len(p["heroes"]) == 1
        assert p["heroes"][0]["hero_name"] == "Reinhardt"
        assert p["heroes"][0]["started_at"] == [0]

    def test_hero_history_with_switches(self):
        history = {
            "PLAYER1": [(100.0, "Reinhardt"), (200.0, "Juno"), (350.0, "Moira")],
        }
        data = self._make_data()
        merge_heroes(data, hero_history=history)
        p = data["players"][0]
        names = [h["hero_name"] for h in p["heroes"]]
        assert "Reinhardt" in names
        assert "Juno" in names
        assert "Moira" in names

    def test_hero_history_started_at_relative(self):
        history = {
            "PLAYER1": [(100.0, "Reinhardt"), (200.0, "Juno")],
        }
        data = self._make_data()
        merge_heroes(data, hero_history=history)
        p = data["players"][0]
        rein = next(h for h in p["heroes"] if h["hero_name"] == "Reinhardt")
        juno = next(h for h in p["heroes"] if h["hero_name"] == "Juno")
        assert rein["started_at"] == [0]
        assert juno["started_at"] == [100]

    def test_analyzer_stats_attached(self):
        hero = {
            "hero_name": "Reinhardt",
            "stats": [{"label": "Charge Kills", "value": "3", "is_featured": True}],
        }
        history = {"PLAYER1": [(100.0, "Reinhardt")]}
        data = self._make_data(hero=hero)
        merge_heroes(data, hero_history=history)
        p = data["players"][0]
        rein = next(h for h in p["heroes"] if h["hero_name"] == "Reinhardt")
        assert len(rein["stats"]) == 1
        assert rein["stats"][0]["label"] == "Charge Kills"

    def test_extra_stats_attached(self):
        extra = [
            {
                "hero_name": "Juno",
                "stats": [{"label": "Weapon Accuracy", "value": "48%", "is_featured": False}],
            }
        ]
        history = {
            "PLAYER1": [(100.0, "Reinhardt"), (200.0, "Juno")],
        }
        data = self._make_data(extra_hero_stats=extra)
        merge_heroes(data, hero_history=history)
        p = data["players"][0]
        juno = next(h for h in p["heroes"] if h["hero_name"] == "Juno")
        assert len(juno["stats"]) == 1
        assert juno["stats"][0]["label"] == "Weapon Accuracy"

    def test_removes_hero_field(self):
        data = self._make_data(hero={"hero_name": "Ana", "stats": []})
        merge_heroes(data)
        for p in data["players"]:
            assert "hero" not in p

    def test_removes_extra_from_toplevel(self):
        data = self._make_data(extra_hero_stats=[])
        merge_heroes(data)
        assert "extra_hero_stats" not in data

    def test_fuzzy_match_hero_names(self):
        """OCR typo in hero_history should still merge with analyzer hero."""
        hero = {
            "hero_name": "Reinhardt",
            "stats": [{"label": "Charge Kills", "value": "3", "is_featured": True}],
        }
        # Subtitle OCR read "Reinhardtt" (extra t)
        history = {"PLAYER1": [(100.0, "Reinhardtt")]}
        data = self._make_data(hero=hero)
        merge_heroes(data, hero_history=history)
        p = data["players"][0]
        # Should merge into one entry, not create two
        assert len(p["heroes"]) == 1
        assert p["heroes"][0]["stats"][0]["label"] == "Charge Kills"

    def test_no_overwrite_main_stats(self):
        """Extra stats should not overwrite existing main hero stats."""
        hero = {
            "hero_name": "Reinhardt",
            "stats": [{"label": "Charge Kills", "value": "3", "is_featured": True}],
        }
        extra = [
            {
                "hero_name": "Reinhardt",
                "stats": [{"label": "Different", "value": "99", "is_featured": False}],
            }
        ]
        history = {"PLAYER1": [(100.0, "Reinhardt")]}
        data = self._make_data(hero=hero, extra_hero_stats=extra)
        merge_heroes(data, hero_history=history)
        p = data["players"][0]
        rein = next(h for h in p["heroes"] if h["hero_name"] in ("Reinhardt", "Reinhardtt"))
        assert rein["stats"][0]["label"] == "Charge Kills"  # not overwritten

    def test_unmatched_analyzer_hero_added(self):
        """Analyzer hero not in history should be added with empty started_at."""
        hero = {
            "hero_name": "Moira",
            "stats": [{"label": "Healing", "value": "5000", "is_featured": True}],
        }
        data = self._make_data(hero=hero)
        merge_heroes(data)
        p = data["players"][0]
        assert len(p["heroes"]) == 1
        assert p["heroes"][0]["hero_name"] == "Moira"
        assert p["heroes"][0]["started_at"] == []

    def test_non_self_player_gets_history(self):
        """Non-self players should also get heroes from hero_history."""
        history = {"PLAYER2": [(100.0, "Genji"), (200.0, "Tracer")]}
        data = self._make_data()
        merge_heroes(data, hero_history=history)
        p2 = next(p for p in data["players"] if p["player_name"] == "PLAYER2")
        assert len(p2["heroes"]) == 2

    def test_hero_switch_back(self):
        """Player switching back to same hero should group started_at times."""
        history = {
            "PLAYER1": [(100.0, "Reinhardt"), (200.0, "Juno"), (300.0, "Reinhardt")],
        }
        data = self._make_data()
        merge_heroes(data, hero_history=history)
        p = data["players"][0]
        rein = next(h for h in p["heroes"] if h["hero_name"] == "Reinhardt")
        assert rein["started_at"] == [0, 200]  # relative to match_start=100


class TestFormatMatch:
    def _make_formatted_data(self, **overrides):
        data = {
            "map_name": "Lijiang Tower",
            "duration": "8:06",
            "mode": "CONTROL",
            "queue_type": "COMPETITIVE",
            "result": "VICTORY",
            "players": [
                {
                    "team": "ALLY", "role": "TANK", "player_name": "PLAYER1",
                    "title": None,
                    "eliminations": 15, "assists": 8, "deaths": 3,
                    "damage": 12500, "healing": 0, "mitigation": 8900,
                    "is_self": True,
                    "heroes": [{"hero_name": "Reinhardt", "started_at": [0], "stats": []}],
                },
                {
                    "team": "ENEMY", "role": "TANK", "player_name": "ENEMY1",
                    "title": None,
                    "eliminations": 10, "assists": 5, "deaths": 6,
                    "damage": 9000, "healing": 0, "mitigation": 7000,
                    "is_self": False, "heroes": [],
                },
            ],
        }
        data.update(overrides)
        return data

    def test_contains_basic_fields(self):
        data = self._make_formatted_data()
        text = format_match(data)
        assert "MAP: Lijiang Tower" in text
        assert "TIME: 8:06" in text
        assert "MODE: CONTROL" in text
        assert "RESULT: VICTORY" in text

    def test_contains_team_headers(self):
        data = self._make_formatted_data()
        text = format_match(data)
        assert "YOUR TEAM" in text
        assert "ENEMY TEAM" in text

    def test_hero_stats_section(self):
        data = self._make_formatted_data()
        data["players"][0]["heroes"][0]["stats"] = [
            {"label": "Charge Kills", "value": "3", "is_featured": True},
        ]
        text = format_match(data)
        assert "HERO STATS:" in text
        assert "Reinhardt" in text
        assert "Charge Kills: 3" in text

    def test_no_hero_stats_when_empty(self):
        data = self._make_formatted_data()
        text = format_match(data)
        assert "HERO STATS:" not in text

    def test_hero_switches_section(self):
        history = {
            "PLAYER1": [(100.0, "Reinhardt"), (200.0, "Juno")],
        }
        data = self._make_formatted_data()
        text = format_match(data, hero_history=history)
        assert "HERO SWITCHES:" in text

    def test_no_switches_when_single_hero(self):
        history = {"PLAYER1": [(100.0, "Reinhardt")]}
        data = self._make_formatted_data()
        text = format_match(data, hero_history=history)
        assert "HERO SWITCHES:" not in text

    def test_player_hero_bracket_from_map(self):
        data = self._make_formatted_data()
        hero_map = {"PLAYER1": "Reinhardt"}
        text = format_match(data, hero_map=hero_map)
        assert "[Reinhardt]" in text

    def test_player_switch_chain(self):
        history = {
            "PLAYER1": [(100.0, "Reinhardt"), (200.0, "Juno"), (300.0, "Moira")],
        }
        data = self._make_formatted_data()
        text = format_match(data, hero_history=history)
        assert "[Reinhardt > Juno > Moira]" in text

    def test_null_stats_formatted_as_dash(self):
        data = self._make_formatted_data()
        data["players"][0]["eliminations"] = None
        text = format_match(data)
        assert "- |" in text


class TestGetRanksReference:
    def test_returns_bytes_when_exists(self):
        result = get_ranks_reference()
        assert isinstance(result, bytes)
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"

    def test_returns_none_when_missing(self, monkeypatch):
        from pathlib import Path
        monkeypatch.setattr(
            "overwatchlooker.analyzers.common._RANKS_REF_PATH",
            Path("/nonexistent/ranks.png"),
        )
        assert get_ranks_reference() is None


class TestLogCost:
    def test_writes_jsonl(self, tmp_path, monkeypatch):
        log_path = tmp_path / "costs.jsonl"
        monkeypatch.setattr("overwatchlooker.analyzers.common._COST_LOG", log_path)
        log_cost("test-model", 100, 50, 0.01)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["model"] == "test-model"
        assert entry["input_tokens"] == 100
        assert entry["cost_usd"] == 0.01

    def test_with_elapsed(self, tmp_path, monkeypatch):
        log_path = tmp_path / "costs.jsonl"
        monkeypatch.setattr("overwatchlooker.analyzers.common._COST_LOG", log_path)
        log_cost("test-model", 100, 50, 0.01, elapsed=2.5)
        entry = json.loads(log_path.read_text().strip())
        assert entry["elapsed_s"] == 2.5

    def test_appends_multiple(self, tmp_path, monkeypatch):
        log_path = tmp_path / "costs.jsonl"
        monkeypatch.setattr("overwatchlooker.analyzers.common._COST_LOG", log_path)
        log_cost("m1", 10, 5, 0.001)
        log_cost("m2", 20, 10, 0.002)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
