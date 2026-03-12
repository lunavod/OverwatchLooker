"""Tests for disk cache: get/put round-trip, key isolation."""

import pytest

from overwatchlooker import cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


class TestCache:
    def test_roundtrip_dict(self, cache_dir):
        data = {"map_name": "Numbani", "result": "VICTORY"}
        cache.put(b"image1", "codex", data)
        result = cache.get(b"image1", "codex")
        assert result == data

    def test_miss(self, cache_dir):
        assert cache.get(b"nonexistent", "codex") is None

    def test_key_isolation_by_analyzer(self, cache_dir):
        cache.put(b"same_image", "codex", {"source": "codex"})
        cache.put(b"same_image", "anthropic", {"source": "anthropic"})
        assert cache.get(b"same_image", "codex")["source"] == "codex"
        assert cache.get(b"same_image", "anthropic")["source"] == "anthropic"

    def test_key_isolation_by_image(self, cache_dir):
        cache.put(b"image_a", "codex", {"id": "a"})
        cache.put(b"image_b", "codex", {"id": "b"})
        assert cache.get(b"image_a", "codex")["id"] == "a"
        assert cache.get(b"image_b", "codex")["id"] == "b"
