import argparse
import io
import sys
from pathlib import Path

from PIL import Image

from overwatchlooker.config import ANALYZER
from overwatchlooker.display import print_analysis, print_error, print_status
from overwatchlooker.notification import copy_to_clipboard, show_notification
from overwatchlooker.tray import App


def main():
    parser = argparse.ArgumentParser(description="Overwatch 2 screen analyzer")
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to an image file to analyze instead of listening for hotkeys",
    )
    parser.add_argument(
        "--analyzer",
        choices=["anthropic", "codex", "ocr"],
        default=None,
        help="Analyzer backend (overrides ANALYZER env var)",
    )
    parser.add_argument(
        "--tg",
        action="store_true",
        help="Send results to Telegram instead of copying to clipboard",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Upload match results to the MCP server",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Use audio-based detection instead of subtitle OCR (requires proc-tap)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Bypass cache and re-analyze from scratch",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Mark match as backfilled when uploading to MCP",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Log subtitle OCR results to transcripts/ folder",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Replay a recording directory instead of live capture",
    )
    # --speed removed: tick-based replay always runs at max speed
    result_group = parser.add_mutually_exclusive_group()
    result_group.add_argument(
        "--win",
        action="store_true",
        help="Hint that the match result is VICTORY",
    )
    result_group.add_argument(
        "--loss",
        action="store_true",
        help="Hint that the match result is DEFEAT",
    )
    args = parser.parse_args()

    # CLI --analyzer overrides env var
    analyzer = args.analyzer or ANALYZER
    if args.analyzer:
        import overwatchlooker.config as cfg
        cfg.ANALYZER = args.analyzer

    features = [f"analyzer={analyzer}"]
    if args.tg:
        features.append("telegram")
    if args.mcp:
        features.append("mcp")
    if args.audio:
        features.append("audio")
    if args.transcript:
        features.append("transcript")
    print_status(f"OverwatchLooker started ({', '.join(features)})")

    if args.image:
        path = Path(args.image)
        if not path.is_file():
            print_error(f"File not found: {path}")
            sys.exit(1)
        img = Image.open(path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        from overwatchlooker import cache

        result = None
        if not args.clean:
            result = cache.get(png_bytes, analyzer)
            if result:
                print_status("Using cached result.")

        if result is None:
            from overwatchlooker.analyzers import get_analyze_screenshot
            analyze_screenshot = get_analyze_screenshot()
            audio_result = "VICTORY" if args.win else "DEFEAT" if args.loss else None
            result = analyze_screenshot(png_bytes, audio_result=audio_result)
            cache.put(png_bytes, analyzer, result)

        # Convert dict (structured output) to display text
        if isinstance(result, dict):
            if result.get("not_ow2_tab"):
                print_error("Image does not appear to be an OW2 Tab screen.")
                sys.exit(1)
            from overwatchlooker.analyzers.common import format_match
            display_text = format_match(result)
        else:
            # Legacy str result (ocr backend or old cache)
            if result.startswith("NOT_OW2_TAB"):
                print_error("Image does not appear to be an OW2 Tab screen.")
                sys.exit(1)
            display_text = result

        formatted = print_analysis(display_text)
        if args.mcp and isinstance(result, dict):
            from overwatchlooker.mcp_client import submit_match
            try:
                submit_match(result, png_bytes=png_bytes, is_backfill=args.backfill)
                print_status("Uploaded to MCP.")
            except Exception as e:
                print_error(f"MCP upload failed: {e}")
        if args.tg:
            from overwatchlooker.telegram import send_message
            if send_message(formatted):
                show_notification("OverwatchLooker", "Analysis sent to Telegram.")
            else:
                print_error("Failed to send to Telegram.")
        else:
            copy_to_clipboard(formatted)
            show_notification("OverwatchLooker", "Analysis complete. Copied to clipboard.")
    elif args.replay:
        from overwatchlooker.recording.replay import ReplaySource

        replay_dir = Path(args.replay)
        if not replay_dir.is_dir():
            print_error(f"Recording directory not found: {replay_dir}")
            sys.exit(1)

        replay = ReplaySource(replay_dir)
        print_status(f"Replaying {replay_dir.name} ({replay.duration:.0f}s, "
                     f"{replay.resolution[0]}x{replay.resolution[1]}, max speed)")

        app = App(use_telegram=args.tg, use_mcp=args.mcp, use_transcript=args.transcript, replay_source=replay)
        app._start_listening()

        try:
            app._tick_loop.run()  # blocks until replay exhausted
        except KeyboardInterrupt:
            if app._tick_loop:
                app._tick_loop.stop()
        finally:
            # Wait for any in-progress analysis to finish
            import time as _time
            for _ in range(60):
                if not app._analyzing:
                    break
                _time.sleep(1.0)
            app._stop_listening()
            replay.close()
            print_status("Replay finished.")
    else:
        app = App(use_telegram=args.tg, use_mcp=args.mcp, use_audio=args.audio, use_transcript=args.transcript)
        app.run()


if __name__ == "__main__":
    main()
