"""Tests for recording uploader cleanup logic."""

from pathlib import Path

from overwatchlooker.recording_uploader import cleanup_old_recordings


def _make_recording(base: Path, name: str, uploaded: bool = False) -> Path:
    d = base / name
    d.mkdir()
    (d / "recording.mp4").write_bytes(b"fake")
    (d / "match_info.json").write_text('{"mcp_id": "x"}')
    if uploaded:
        (d / ".uploaded").write_text("2026-01-01T00:00:00Z")
    return d


class TestCleanupOldRecordings:
    def test_deletes_uploaded_beyond_keep(self, tmp_path):
        _make_recording(tmp_path, "2026-01-01_01", uploaded=True)
        _make_recording(tmp_path, "2026-01-02_01", uploaded=True)
        _make_recording(tmp_path, "2026-01-03_01", uploaded=True)
        cleanup_old_recordings(tmp_path, keep=2)
        remaining = sorted(d.name for d in tmp_path.iterdir())
        assert remaining == ["2026-01-02_01", "2026-01-03_01"]

    def test_keeps_non_uploaded(self, tmp_path):
        _make_recording(tmp_path, "2026-01-01_01", uploaded=False)
        _make_recording(tmp_path, "2026-01-02_01", uploaded=True)
        _make_recording(tmp_path, "2026-01-03_01", uploaded=True)
        cleanup_old_recordings(tmp_path, keep=1)
        remaining = sorted(d.name for d in tmp_path.iterdir())
        # Newest kept, oldest not uploaded so kept, middle uploaded+beyond keep so deleted
        assert "2026-01-03_01" in remaining
        assert "2026-01-01_01" in remaining  # not uploaded, never deleted
        assert "2026-01-02_01" not in remaining

    def test_no_delete_within_keep(self, tmp_path):
        _make_recording(tmp_path, "2026-01-01_01", uploaded=True)
        _make_recording(tmp_path, "2026-01-02_01", uploaded=True)
        cleanup_old_recordings(tmp_path, keep=5)
        remaining = sorted(d.name for d in tmp_path.iterdir())
        assert len(remaining) == 2

    def test_keep_zero_does_nothing(self, tmp_path):
        _make_recording(tmp_path, "2026-01-01_01", uploaded=True)
        cleanup_old_recordings(tmp_path, keep=0)
        assert len(list(tmp_path.iterdir())) == 1

    def test_empty_dir(self, tmp_path):
        cleanup_old_recordings(tmp_path, keep=5)  # should not raise

    def test_nonexistent_dir(self, tmp_path):
        cleanup_old_recordings(tmp_path / "nope", keep=5)  # should not raise
