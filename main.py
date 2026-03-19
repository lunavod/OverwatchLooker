import argparse
import io
import sys
from pathlib import Path

from PIL import Image

from overwatchlooker.config import ANALYZER
from overwatchlooker.display import print_analysis, print_error, print_status
from overwatchlooker.notification import show_notification
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
        choices=["anthropic", "codex"],
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
        help="Replay a recording directory or .mp4 file",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip LLM analysis on detection (useful for testing replays)",
    )
    parser.add_argument(
        "--ws",
        action="store_true",
        help="Start WebSocket server for companion app",
    )
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

    # Start WebSocket server if requested
    event_bus = None
    ws_server = None
    if args.ws:
        from overwatchlooker.ws_server import EventBus, WsServer
        from overwatchlooker.config import WS_PORT
        event_bus = EventBus()
        ws_server = WsServer(event_bus, port=WS_PORT)
        ws_server.start()

    features = [f"analyzer={analyzer}"]
    if args.tg:
        features.append("telegram")
    if args.mcp:
        features.append("mcp")
    if args.transcript:
        features.append("transcript")
    if args.ws:
        from overwatchlooker.config import WS_PORT
        features.append(f"ws://0.0.0.0:{WS_PORT}")
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
            show_notification("OverwatchLooker", "Analysis complete.")
    elif args.replay:
        from overwatchlooker.recording.replay import ReplaySource

        source = Path(args.replay)
        if not source.exists():
            print_error(f"Recording not found: {source}")
            sys.exit(1)

        replay = ReplaySource(source)
        print_status(f"Replaying {source.name} ({replay.duration:.0f}s, "
                     f"{replay.resolution[0]}x{replay.resolution[1]}, max speed)")

        app = App(use_telegram=args.tg, use_mcp=args.mcp, use_transcript=args.transcript,
                  replay_source=replay, no_analysis=args.no_analysis, event_bus=event_bus)
        app._start_listening()

        try:
            app._tick_loop.run()  # blocks until replay exhausted
        except KeyboardInterrupt:
            if app._tick_loop:
                app._tick_loop.stop()
        finally:
            # Flush any pending detection that didn't fire before replay ended
            if hasattr(app, '_subtitle_system') and app._subtitle_system:
                final_sim_time = replay.frame_count / replay.fps
                app._subtitle_system.flush_pending(final_sim_time)

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
        app = App(use_telegram=args.tg, use_mcp=args.mcp, use_transcript=args.transcript,
                  event_bus=event_bus)
        app.run()


if __name__ == "__main__":
    main()
