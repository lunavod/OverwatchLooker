"""Tests for llm_analyzer module: schema validation, response parsing."""

import json

from overwatchlooker.llm_analyzer import MATCH_SCHEMA


class TestMatchSchema:
    def test_schema_has_required_fields(self):
        props = MATCH_SCHEMA["schema"]["properties"]
        assert "not_ow2_tab" in props
        assert "map_name" in props
        assert "duration" in props
        assert "mode" in props
        assert "queue_type" in props
        assert "players" in props

    def test_player_schema_has_required_fields(self):
        player_props = MATCH_SCHEMA["schema"]["properties"]["players"]["items"]["properties"]
        for field in ["team", "role", "player_name", "eliminations", "assists",
                      "deaths", "damage", "healing", "mitigation", "is_self",
                      "hero_name", "title"]:
            assert field in player_props, f"Missing player field: {field}"

    def test_schema_is_valid_json(self):
        # Ensure the schema can be serialized (required for API call)
        serialized = json.dumps(MATCH_SCHEMA)
        parsed = json.loads(serialized)
        assert parsed["name"] == "match_analysis"

    def test_mode_enum_values(self):
        mode_enum = MATCH_SCHEMA["schema"]["properties"]["mode"]["enum"]
        assert set(mode_enum) == {"PUSH", "CONTROL", "ESCORT", "HYBRID", "CLASH", "FLASHPOINT"}

    def test_queue_type_enum_values(self):
        qt_enum = MATCH_SCHEMA["schema"]["properties"]["queue_type"]["enum"]
        assert set(qt_enum) == {"COMPETITIVE", "QUICKPLAY"}
