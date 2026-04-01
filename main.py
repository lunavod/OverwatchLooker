import argparse
import sys
from pathlib import Path

from overwatchlooker.display import print_error, print_status
from overwatchlooker.tray import App


def main():
    parser = argparse.ArgumentParser(description="Overwatch 2 match tracker")
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
        "--replay-duration",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Only replay the first N seconds of the recording",
    )
    parser.add_argument(
        "--replay-start",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Start replay from N seconds into the recording",
    )
    parser.add_argument(
        "--ws",
        action="store_true",
        help="Start WebSocket server for companion app",
    )
    parser.add_argument(
        "--overwolf",
        action="store_true",
        help="Start Overwolf GEP receiver (accepts OverwatchListener connections)",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Submit completed matches to the MCP server",
    )
    parser.add_argument(
        "--auto-recording",
        action="store_true",
        help="Automatically record matches (start on match start, stop after match end)",
    )
    parser.add_argument(
        "--auto-recording-tail",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Seconds to keep recording after match ends (default: 60)",
    )
    args = parser.parse_args()

    # Start WebSocket server if requested
    event_bus = None
    ws_server = None
    if args.ws:
        from overwatchlooker.ws_server import EventBus, WsServer
        from overwatchlooker.config import WS_PORT
        event_bus = EventBus()
        ws_server = WsServer(event_bus, port=WS_PORT)
        ws_server.start()

    # Start Overwolf receiver if requested
    overwolf_receiver = None
    if args.overwolf:
        from overwatchlooker.overwolf import OverwolfReceiver
        from overwatchlooker.config import OVERWOLF_PORT
        overwolf_receiver = OverwolfReceiver(port=OVERWOLF_PORT)
        overwolf_receiver.start()

    features = []
    if args.transcript:
        features.append("transcript")
    if args.ws:
        from overwatchlooker.config import WS_PORT
        features.append(f"ws://0.0.0.0:{WS_PORT}")
    if args.overwolf:
        from overwatchlooker.config import OVERWOLF_PORT
        features.append(f"overwolf://0.0.0.0:{OVERWOLF_PORT}")
    if args.mcp:
        features.append("mcp")
    if args.auto_recording:
        features.append(f"auto-recording (tail={args.auto_recording_tail}s)")
    print_status(f"OverwatchLooker started ({', '.join(features)})")

    if args.replay:
        from overwatchlooker.recording.replay import ReplaySource

        source = Path(args.replay)
        if not source.exists():
            print_error(f"Recording not found: {source}")
            sys.exit(1)

        replay = ReplaySource(source)
        print_status(f"Replaying {source.name} ({replay.duration:.0f}s, "
                     f"{replay.resolution[0]}x{replay.resolution[1]}, max speed)")

        app = App(use_transcript=args.transcript,
                  replay_source=replay, event_bus=event_bus,
                  overwolf_receiver=overwolf_receiver,
                  use_mcp=args.mcp,
                  auto_recording=args.auto_recording,
                  auto_recording_tail=args.auto_recording_tail)
        app._start_listening()

        if args.replay_start and app._tick_loop:
            app._tick_loop.start_tick = int(args.replay_start * replay.fps)
        if args.replay_duration and app._tick_loop:
            start = app._tick_loop.start_tick
            app._tick_loop.max_ticks = start + int(args.replay_duration * replay.fps)

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

            app.wait_for_analysis()
            app._stop_listening()
            replay.close()
            print_status("Replay finished.")
    else:
        app = App(use_transcript=args.transcript,
                  event_bus=event_bus, overwolf_receiver=overwolf_receiver,
                  use_mcp=args.mcp,
                  auto_recording=args.auto_recording,
                  auto_recording_tail=args.auto_recording_tail)
        app.run()


if __name__ == "__main__":
    main()
