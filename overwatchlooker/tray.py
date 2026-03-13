import logging
import threading
import traceback

import pystray
from PIL import Image, ImageDraw, ImageFont

from overwatchlooker.config import ANALYZER, SCREENSHOT_MAX_AGE_SECONDS
from overwatchlooker.display import print_analysis, print_error, print_status

_logger = logging.getLogger("overwatchlooker")
from overwatchlooker.notification import copy_to_clipboard, show_notification
from overwatchlooker.screenshot import (
    crop_hero_panel,
    has_hero_panel,
    ocr_hero_name,
)
from overwatchlooker.heroes import edit_distance as _edit_distance


def _create_icon_image() -> Image.Image:
    """Create a simple 64x64 icon: blue circle with 'OW' text."""
    img = Image.new("RGB", (64, 64), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=(51, 153, 255))
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((14, 18), "OW", fill="white", font=font)
    return img


_AUDIO_ANALYSIS_DELAY = 5.0  # seconds to wait after audio trigger before analyzing
_TAB_DEBOUNCE = 1.5  # ignore Tab presses within this window of each other


class App:
    def __init__(self, use_telegram: bool = False, use_mcp: bool = False,
                 use_transcript: bool = False, replay_source=None,
                 no_analysis: bool = False):
        self._active = False
        self._detector = None  # SubtitleSystem
        self._analyzing = False
        self._lock = threading.Lock()
        self._valid_tabs: list[tuple[bytes, float, str]] = []  # last 2 valid (png_bytes, timestamp, filename)
        self._hero_crops: dict[str, bytes] = {}  # hero_name -> cropped PNG bytes
        self._use_telegram = use_telegram
        self._use_mcp = use_mcp
        self._use_transcript = use_transcript
        self._replay_source = replay_source  # ReplaySource for replay mode
        self._no_analysis = no_analysis
        self._recorder = None  # Recorder for recording mode
        self._tick_loop = None  # TickLoop instance

    def store_valid_tab(self, png_bytes: bytes, timestamp: float, filename: str) -> None:
        """Store a valid Tab screenshot (called by TabCaptureSystem).

        timestamp is time.monotonic() in live mode, sim_time in replay mode.
        """
        with self._lock:
            self._valid_tabs.append((png_bytes, timestamp, filename))
            if len(self._valid_tabs) > 2:
                self._valid_tabs.pop(0)
        print_status(f"Tab screenshot saved: {filename}")

    def store_hero_crop(self, name: str, crop: bytes) -> None:
        """Store a hero panel crop (called by TabCaptureSystem)."""
        with self._lock:
            if not any(_edit_distance(name.lower(), k.lower()) <= 2
                       for k in self._hero_crops):
                self._hero_crops[name] = crop
                _logger.info(f"Stored hero crop: {name}")
            else:
                _logger.debug(f"Hero crop dedup skip: {name}")

    def _on_detected(self, result: str, detection_time: float = 0.0) -> None:
        """Immediate callback when VICTORY/DEFEAT is first detected."""
        if self._recorder:
            self._recorder.log_event("detection", result=result)
        show_notification("OverwatchLooker", f"{result} detected! Analyzing...")
        if self._no_analysis:
            print_status(f"Detected: {result} (analysis skipped)")
        else:
            print_status(f"Detected: {result}. Analyzing after delay...")

    def _on_detection(self, result: str, detection_time: float = 0.0) -> None:
        """Delayed callback: start analysis after the tick-based delay has elapsed."""
        if self._no_analysis:
            return
        with self._lock:
            if self._analyzing:
                print_status("Analysis already in progress, ignoring detection trigger.")
                return
            self._analyzing = True
            hero_crops = dict(self._hero_crops)
            self._hero_crops.clear()

        # Grab hero map and history from subtitle listener before resetting
        hero_map = {}
        hero_history = {}
        if hasattr(self._detector, "hero_map"):
            hero_map = self._detector.hero_map
        if hasattr(self._detector, "hero_history"):
            hero_history = self._detector.hero_history
        if hasattr(self._detector, "reset_match"):
            self._detector.reset_match()

        thread = threading.Thread(target=self._run_analysis,
                                  args=(result, detection_time, hero_map, hero_history, hero_crops), daemon=True)
        thread.start()

    def _run_analysis(self, detection_result: str, detection_time: float = 0.0,
                      hero_map: dict[str, str] | None = None,
                      hero_history: dict[str, list[tuple[float, str]]] | None = None,
                      hero_crops: dict[str, bytes] | None = None) -> None:
        """Wait, then analyze last valid tab screenshot (fall back to previous if rejected)."""
        try:
            print_status(f"Analyzing {detection_result}...")

            with self._lock:
                tabs = list(self._valid_tabs)

            if not tabs:
                print_error("No valid Tab screenshot found. Press Tab during the scoreboard first.")
                show_notification("OverwatchLooker", "No Tab screenshot to analyze.")
                return

            # Try most recent first, then fall back to previous
            for i, (png_bytes, tab_time, filename) in enumerate(reversed(tabs)):
                age = detection_time - tab_time
                if age > SCREENSHOT_MAX_AGE_SECONDS:
                    continue

                is_fallback = i > 0
                if is_fallback:
                    time_diff = tabs[-1][1] - tab_time
                    print_status(f"Latest screenshot rejected by analyzer. "
                                 f"Falling back to previous ({time_diff:.0f}s older).")

                print_status(f"Analyzing {filename} ({len(png_bytes)} bytes, "
                             f"{age:.0f}s old) with {ANALYZER} backend...")

                # Remove the final screenshot's hero from crops to avoid duplicate
                crops_for_analyzer = dict(hero_crops) if hero_crops else {}
                if crops_for_analyzer and has_hero_panel(png_bytes):
                    final_crop = crop_hero_panel(png_bytes)
                    final_hero = ocr_hero_name(final_crop)
                    if final_hero:
                        to_remove = [k for k in crops_for_analyzer
                                     if _edit_distance(final_hero.lower(), k.lower()) <= 2]
                        for k in to_remove:
                            del crops_for_analyzer[k]
                            _logger.info(f"Removed '{k}' from hero crops (matches final hero '{final_hero}')")

                if crops_for_analyzer:
                    _logger.info(f"Sending {len(crops_for_analyzer)} extra hero crops: {list(crops_for_analyzer.keys())}")

                from overwatchlooker.analyzers import get_analyze_screenshot
                analyze_screenshot = get_analyze_screenshot()
                result = analyze_screenshot(png_bytes, audio_result=detection_result,
                                            hero_crops=crops_for_analyzer or None)

                if result.get("not_ow2_tab"):
                    print_status("Analyzer rejected screenshot as not OW2 Tab.")
                    continue
                # Override UNKNOWN result with subtitle detection
                if result.get("result") == "UNKNOWN" and detection_result:
                    result["result"] = detection_result
                # Merge hero_history + analyzer hero + extra_hero_stats into per-player heroes[]
                from overwatchlooker.analyzers.common import merge_heroes, format_match
                merge_heroes(result, hero_map=hero_map, hero_history=hero_history)
                display_text = format_match(result, hero_map=hero_map,
                                            hero_history=hero_history)

                formatted = print_analysis(display_text)
                if self._use_mcp:
                    from overwatchlooker.mcp_client import submit_match
                    try:
                        submit_match(result, png_bytes=png_bytes)
                        print_status("Uploaded to MCP.")
                    except Exception as e:
                        print_error(f"MCP upload failed: {e}")
                if is_fallback:
                    notif_msg = (f"{detection_result} — Used fallback screenshot "
                                 f"({time_diff:.0f}s older, latest was rejected).")
                else:
                    notif_msg = detection_result
                if self._use_telegram:
                    from overwatchlooker.telegram import send_message
                    if send_message(formatted):
                        show_notification("OverwatchLooker", f"{notif_msg} — Sent to Telegram.")
                    else:
                        print_error("Failed to send to Telegram.")
                        copy_to_clipboard(formatted)
                else:
                    copy_to_clipboard(formatted)
                    show_notification("OverwatchLooker", f"{notif_msg} — Copied to clipboard.")
                return

            # All screenshots were rejected or too old
            print_error("No usable Tab screenshot found.")
            show_notification("OverwatchLooker", "All Tab screenshots rejected or too old.")
        except Exception as e:
            print_error(f"Analysis failed: {e}")
        finally:
            with self._lock:
                self._analyzing = False

    def _start_listening(self) -> None:
        if self._active:
            return

        if ANALYZER == "anthropic":
            from overwatchlooker.config import ANTHROPIC_API_KEY
            if not ANTHROPIC_API_KEY:
                print_error("ANALYZER=anthropic but ANTHROPIC_API_KEY is not set. "
                            "Set it in .env or use a different analyzer.")
                return

        self._active = True
        if self._use_telegram:
            from overwatchlooker.config import TELEGRAM_API_HASH, TELEGRAM_API_ID, TELEGRAM_CHANNEL
            if TELEGRAM_API_ID and TELEGRAM_API_HASH and TELEGRAM_CHANNEL:
                print_status(f"Telegram: ON (chat_id={TELEGRAM_CHANNEL})")
            else:
                print_error("Telegram: --tg flag set but TELEGRAM_API_ID, TELEGRAM_API_HASH, or TELEGRAM_CHANNEL missing in .env")
        else:
            print_status("Telegram: OFF")

        from overwatchlooker.tick import (
            LiveFrameSource, LiveInputSource, ReplayFrameSource, ReplayInputSource,
            SubtitleSystem, TabCaptureSystem, TickLoop,
        )
        from overwatchlooker.config import SUBTITLE_POLL_INTERVAL

        if self._replay_source:
            fps = self._replay_source.fps
            frame_source = ReplayFrameSource(self._replay_source.reader)
            input_source = ReplayInputSource(self._replay_source.events)
            detect_mode = "replay"
        else:
            fps = 10
            frame_source = LiveFrameSource(fps)
            input_source = LiveInputSource()
            detect_mode = "subtitle"

        self._tick_loop = TickLoop(fps, frame_source, input_source)

        tab_system = TabCaptureSystem(self, fps=fps)
        self._tick_loop.register(tab_system.on_tick, every_n_ticks=1)

        subtitle_interval = max(1, int(fps * SUBTITLE_POLL_INTERVAL))
        detection_delay = int(fps * _AUDIO_ANALYSIS_DELAY)
        subtitle_system = SubtitleSystem(on_match=self._on_detection,
                                         on_detected=self._on_detected,
                                         transcript=self._use_transcript,
                                         detection_delay_ticks=detection_delay)
        self._tick_loop.register(subtitle_system.on_tick, every_n_ticks=subtitle_interval)
        self._detector = subtitle_system
        self._subtitle_system = subtitle_system

        if not self._replay_source:
            # Live mode: run tick loop in daemon thread
            tick_thread = threading.Thread(target=self._tick_loop.run, daemon=True)
            tick_thread.start()

        print_status(f"Listening (analyzer={ANALYZER}, detection={detect_mode}). Tab=screenshot.")

    def _stop_listening(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._tick_loop:
            self._tick_loop.stop()
            self._tick_loop = None
        if hasattr(self, '_subtitle_system') and self._subtitle_system:
            self._subtitle_system.close()
            self._subtitle_system = None
        if self._detector and hasattr(self._detector, 'stop'):
            self._detector.stop()
        self._detector = None
        print_status("Stopped listening.")

    def _on_submit_tab(self, result: str) -> None:
        """Manually submit last tab screenshot with a given result."""
        with self._lock:
            if self._analyzing:
                print_status("Analysis already in progress.")
                return
            self._analyzing = True
            hero_crops = dict(self._hero_crops)
            self._hero_crops.clear()

        hero_map = {}
        hero_history = {}
        if hasattr(self._detector, "hero_map"):
            hero_map = self._detector.hero_map
        if hasattr(self._detector, "hero_history"):
            hero_history = self._detector.hero_history
        if hasattr(self._detector, "reset_match"):
            self._detector.reset_match()

        detection_time = self._tick_loop._current_tick / self._tick_loop.fps if self._tick_loop else 0.0
        thread = threading.Thread(target=self._run_analysis,
                                  args=(result, detection_time, hero_map, hero_history, hero_crops), daemon=True)
        thread.start()

    def _on_submit_win(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_submit_tab("VICTORY")

    def _on_submit_loss(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_submit_tab("DEFEAT")

    def _on_toggle_recording(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Toggle screen recording on/off."""
        if self._recorder and self._recorder.is_recording:
            try:
                output = self._recorder.stop()
                if self._tick_loop:
                    self._tick_loop.on_frame = None
                    self._tick_loop.on_key_events = None
                self._recorder = None
                print_status(f"Recording saved to {output}")
                show_notification("OverwatchLooker", f"Recording saved.")
            except Exception as e:
                print_error(f"Failed to stop recording: {e}")
        else:
            try:
                from overwatchlooker.recording.recorder import Recorder
                self._recorder = Recorder()
                # Get resolution from the last captured frame or use a default
                resolution = (3840, 2160)
                if self._tick_loop and self._tick_loop.frame_source:
                    src = self._tick_loop.frame_source
                    if hasattr(src, '_camera') and src._camera:
                        import dxcam
                        frame = src._camera.grab()
                        if frame is not None:
                            h, w = frame.shape[:2]
                            resolution = (w, h)
                output = self._recorder.start(resolution)
                if self._tick_loop:
                    self._tick_loop.on_frame = self._recorder.push_frame
                    self._tick_loop.on_key_events = self._recorder.log_key_events
                print_status(f"Recording to {output}")
                show_notification("OverwatchLooker", "Recording started.")
            except Exception as e:
                print_error(f"Failed to start recording: {e}")
                self._recorder = None
        self._rebuild_menu()

    def _on_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        if self._recorder and self._recorder.is_recording:
            try:
                if self._tick_loop:
                    self._tick_loop.on_frame = None
                    self._tick_loop.on_key_events = None
                self._recorder.stop()
            except Exception:
                pass
            self._recorder = None
        self._stop_listening()
        icon.stop()

    def _on_toggle(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        try:
            _logger.debug(f"Toggle clicked, active={self._active}")
            if self._active:
                self._stop_listening()
            else:
                self._start_listening()
            _logger.debug(f"After toggle, active={self._active}")
            self._rebuild_menu()
        except Exception:
            _logger.error(f"Toggle error:\n{traceback.format_exc()}")

    def _rebuild_menu(self) -> None:
        """Rebuild the tray menu to reflect current state."""
        label = "Stop Listening" if self._active else "Start Listening"
        rec_label = "Stop Recording" if (self._recorder and self._recorder.is_recording) else "Start Recording"
        self._icon.menu = pystray.Menu(
            pystray.MenuItem(label, self._on_toggle, default=True),
            pystray.MenuItem(rec_label, self._on_toggle_recording),
            pystray.MenuItem("Submit last tab (win)", self._on_submit_win),
            pystray.MenuItem("Submit last tab (loss)", self._on_submit_loss),
            pystray.MenuItem("Quit", self._on_quit),
        )
        self._icon.update_menu()

    def run(self) -> None:
        """Main entry point -- blocks on the tray icon run loop."""
        self._icon = pystray.Icon(
            name="OverwatchLooker",
            icon=_create_icon_image(),
            title="OverwatchLooker",
            menu=pystray.Menu(
                pystray.MenuItem("Stop Listening", self._on_toggle, default=True),
                pystray.MenuItem("Start Recording", self._on_toggle_recording),
                pystray.MenuItem("Submit last tab (win)", self._on_submit_win),
                pystray.MenuItem("Submit last tab (loss)", self._on_submit_loss),
                pystray.MenuItem("Quit", self._on_quit),
            ),
        )

        def setup(icon: pystray.Icon):
            try:
                icon.visible = True
                self._start_listening()
                _logger.debug(f"Setup complete, active={self._active}")
            except Exception:
                _logger.error(f"Setup error:\n{traceback.format_exc()}")

        print_status("OverwatchLooker starting. Check system tray. Press Ctrl+C to quit.")

        # Run tray icon in a daemon thread so the main thread can catch Ctrl+C
        tray_thread = threading.Thread(
            target=self._icon.run, kwargs={"setup": setup}, daemon=True
        )
        tray_thread.start()

        try:
            while tray_thread.is_alive():
                tray_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            self._shutdown()

    def _shutdown(self) -> None:
        print_status("Shutting down...")
        if self._recorder and self._recorder.is_recording:
            try:
                self._recorder.stop()
                print_status("Recording saved on shutdown.")
            except Exception:
                pass
        self._stop_listening()
        if self._icon:
            self._icon.stop()
