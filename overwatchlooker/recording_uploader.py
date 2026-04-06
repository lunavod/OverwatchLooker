"""Background recording uploader via tus resumable upload.

Can run standalone (``python -m overwatchlooker.recording_uploader``) or be
started as a background thread inside the main app.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from tusclient import client as tus_client
from tusclient.storage import filestorage

_SCAN_INTERVAL = 60  # seconds between scans
_MAX_SCAN_INTERVAL = 600  # max backoff when all uploads fail
_RECORDING_FILES = ["recording.mp4", "recording.meta", "recording.overwolf.jsonl"]
_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB

_logger = logging.getLogger("overwatchlooker.uploader")


def _setup_file_logger(log_dir: Path) -> None:
    """Add a file handler to the uploader logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "uploader.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


def upload_recording_dir(
    recording_dir: Path,
    match_id: str,
    upload_url: str,
    auth_key: str,
    cache_file: Path,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> None:
    """Upload all recording files in *recording_dir* for *match_id*.

    Args:
        progress_cb: Optional ``(filename, sent_bytes, total_bytes)`` callback.
    """
    files = [recording_dir / f for f in _RECORDING_FILES
             if (recording_dir / f).exists()]
    if not files:
        _logger.warning(f"No recording files in {recording_dir}")
        return

    tus = tus_client.TusClient(
        upload_url,
        headers={"Authorization": f"Bearer {auth_key}"},
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    storage = filestorage.FileStorage(str(cache_file))

    for filepath in files:
        total = filepath.stat().st_size
        total_mb = total / (1024 * 1024)
        _logger.info(f"Uploading {filepath.name} ({total_mb:.1f} MB)")

        uploader = tus.uploader(
            str(filepath),
            chunk_size=_CHUNK_SIZE,
            metadata={
                "match_id": match_id,
                "filename": filepath.name,
            },
            store_url=True,
            url_storage=storage,
        )

        start_time = time.monotonic()
        start_offset = uploader.offset

        while uploader.offset < total:
            uploader.upload_chunk()
            if progress_cb:
                progress_cb(filepath.name, uploader.offset, total)

        elapsed = time.monotonic() - start_time
        uploaded_mb = (uploader.offset - start_offset) / (1024 * 1024)
        speed = uploaded_mb / elapsed if elapsed > 0 else 0
        _logger.info(
            f"Done: {filepath.name} — {uploaded_mb:.1f} MB in "
            f"{int(elapsed)}s ({speed:.1f} MB/s)")

    # Mark as uploaded
    marker = recording_dir / ".uploaded"
    marker.write_text(
        datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    _logger.info(f"Marked {recording_dir.name} as uploaded")


def _find_pending(recordings_dir: Path) -> list[tuple[Path, str]]:
    """Return list of ``(recording_dir, mcp_id)`` for dirs needing upload."""
    pending: list[tuple[Path, str]] = []
    if not recordings_dir.is_dir():
        return pending
    for d in sorted(recordings_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / ".uploaded").exists():
            continue
        info_path = d / "match_info.json"
        if not info_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mcp_id = info.get("mcp_id")
        if not mcp_id:
            continue
        if not info.get("recording_complete"):
            continue
        pending.append((d, mcp_id))
    return pending


def run_scan_loop(
    recordings_dir: Path,
    upload_url: str,
    auth_key: str,
    cache_file: Path,
    stop_event: threading.Event | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> None:
    """Scan *recordings_dir* every minute and upload pending recordings."""
    _logger.info(f"Upload scanner started — watching {recordings_dir}")
    consecutive_failures = 0
    while True:
        if stop_event and stop_event.is_set():
            break
        try:
            pending = _find_pending(recordings_dir)
            if pending:
                _logger.info(f"Found {len(pending)} pending upload(s)")
            any_success = False
            for rec_dir, mcp_id in pending:
                if stop_event and stop_event.is_set():
                    break
                _logger.info(f"Uploading {rec_dir.name} (match {mcp_id})")
                try:
                    upload_recording_dir(
                        rec_dir, mcp_id, upload_url, auth_key,
                        cache_file, progress_cb=progress_cb)
                    any_success = True
                except Exception as exc:
                    _logger.exception(f"Failed to upload {rec_dir.name}")
                    # Log response body for TUS errors (useful for debugging)
                    if hasattr(exc, "response_content") and exc.response_content:
                        _logger.error(f"Server response: {exc.response_content}")
            if pending:
                if any_success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
        except Exception:
            _logger.exception("Error during upload scan")
            consecutive_failures += 1

        # Exponential backoff when all uploads keep failing
        delay = min(_SCAN_INTERVAL * (2 ** consecutive_failures),
                    _MAX_SCAN_INTERVAL)
        if consecutive_failures > 0:
            _logger.info(f"All uploads failed ({consecutive_failures}x), "
                         f"next scan in {delay}s")

        # Wait for next scan (interruptible)
        if stop_event:
            stop_event.wait(delay)
        else:
            time.sleep(delay)


def start_background(
    recordings_dir: Path,
    log_dir: Path | None = None,
) -> tuple[threading.Thread, threading.Event]:
    """Start the upload scanner as a background daemon thread.

    Returns ``(thread, stop_event)`` so the caller can stop it later.
    """
    from dotenv import load_dotenv
    load_dotenv()

    upload_url = os.environ.get("TUSD_UPLOAD_URL", "")
    auth_key = os.environ.get("TUSD_AUTH_KEY", "")
    if not upload_url or not auth_key:
        _logger.warning("TUSD_UPLOAD_URL / TUSD_AUTH_KEY not set — uploader disabled")
        evt = threading.Event()
        evt.set()
        return threading.Thread(), evt

    if log_dir:
        _setup_file_logger(log_dir)

    cache_file = recordings_dir.parent / ".upload_cache" / "uploads.json"

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_scan_loop,
        args=(recordings_dir, upload_url, auth_key, cache_file, stop_event),
        daemon=True,
        name="recording-uploader",
    )
    thread.start()
    return thread, stop_event


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan recordings_auto/ and upload pending recordings")
    parser.add_argument(
        "--dir", type=Path, default=Path(__file__).parent.parent / "recordings_auto",
        help="Recordings directory to scan (default: recordings_auto/)")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single scan then exit (don't loop)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    upload_url = os.environ.get("TUSD_UPLOAD_URL", "")
    auth_key = os.environ.get("TUSD_AUTH_KEY", "")
    if not upload_url or not auth_key:
        print("Error: TUSD_UPLOAD_URL and TUSD_AUTH_KEY must be set in .env",
              file=__import__("sys").stderr)
        raise SystemExit(1)

    log_dir = Path(__file__).parent.parent / "logs"
    _setup_file_logger(log_dir)
    # Also log to console when running standalone
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logger.addHandler(console)

    cache_file = args.dir.parent / ".upload_cache" / "uploads.json"

    def _progress(filename: str, sent: int, total: int) -> None:
        pct = sent / total * 100
        sent_mb = sent / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        print(f"\r  {filename}: {sent_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)",
              end="", flush=True)

    if args.once:
        pending = _find_pending(args.dir)
        if not pending:
            print("No pending uploads found.")
            return
        for rec_dir, mcp_id in pending:
            print(f"Uploading {rec_dir.name} (match {mcp_id})")
            upload_recording_dir(
                rec_dir, mcp_id, upload_url, auth_key,
                cache_file, progress_cb=_progress)
            print()
        print("Done.")
    else:
        print(f"Watching {args.dir} for pending uploads (Ctrl+C to stop)")
        try:
            run_scan_loop(
                args.dir, upload_url, auth_key, cache_file,
                progress_cb=_progress)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
