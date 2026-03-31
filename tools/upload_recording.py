"""Upload a recording directory to the MCP server via tus resumable upload.

Usage:
    python -m tools.upload_recording <recording_dir> <match_id>

Reads TUSD_UPLOAD_URL and TUSD_AUTH_KEY from .env.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from overwatchlooker.recording_uploader import upload_recording_dir


def _progress(filename: str, sent: int, total: int) -> None:
    pct = sent / total * 100
    sent_mb = sent / (1024 * 1024)
    total_mb = total / (1024 * 1024)
    print(f"\r  {filename}: {sent_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)",
          end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a recording to the MCP server")
    parser.add_argument("recording_dir", type=Path, help="Path to the recording directory")
    parser.add_argument("match_id", help="UUID of the match to attach files to")
    args = parser.parse_args()

    load_dotenv()

    upload_url = os.environ.get("TUSD_UPLOAD_URL", "")
    auth_key = os.environ.get("TUSD_AUTH_KEY", "")

    if not upload_url:
        print("Error: TUSD_UPLOAD_URL not set in .env", file=sys.stderr)
        sys.exit(1)
    if not auth_key:
        print("Error: TUSD_AUTH_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    if not args.recording_dir.is_dir():
        print(f"Error: {args.recording_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    cache_file = Path(__file__).resolve().parent.parent / ".upload_cache" / "uploads.json"

    print(f"Uploading {args.recording_dir} for match {args.match_id}")
    upload_recording_dir(
        args.recording_dir, args.match_id, upload_url, auth_key,
        cache_file, progress_cb=_progress)
    print(f"\nAll files uploaded for match {args.match_id}")


if __name__ == "__main__":
    main()
