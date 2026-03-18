from __future__ import annotations

import logging
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pystray  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFont

from overwatchlooker.config import ANALYZER, SCREENSHOT_MAX_AGE_SECONDS
from overwatchlooker.display import print_analysis, print_error, print_status

if TYPE_CHECKING:
    from memoir_capture import CaptureEngine
    from overwatchlooker.tick import ChatSystem, SubtitleSystem, TickLoop
    from overwatchlooker.ws_server import EventBus

_logger = logging.getLogger("overwatchlooker")
from overwatchlooker.notification import copy_to_clipboard, show_notification  # noqa: E402
from overwatchlooker.screenshot import (  # noqa: E402
    crop_hero_panel,
    has_hero_panel,
    ocr_hero_name,
)
from overwatchlooker.heroes import edit_distance as _edit_distance  # noqa: E402

_RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"



def _create_icon_image() -> Image.Image:
    """Create a simple 64x64 icon: blue circle with 'OW' text."""
    img = Image.new("RGB", (64, 64), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=(51, 153, 255))
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((14, 18), "OW", fill="white", font=font)
    return img


_AUDIO_ANALYSIS_DELAY = 5.0  # seconds to wait after audio trigger before analyzing
_TAB_DEBOUNCE = 1.5  # ignore Tab presses within this window of each other
_POST_SUBMIT_COOLDOWN = 30.0  # seconds to ignore tab/crop events after detection
_OW_POLL_INTERVAL = 2.0  # seconds between OW window checks
_OW_STABILITY_TIME = 5.0  # seconds OW must be present before starting engine


class App:
    def __init__(self, use_telegram: bool = False, use_mcp: bool = False,
                 use_transcript: bool = False, replay_source=None,
                 no_analysis: bool = False, event_bus: EventBus | None = None):
        self._active = False
        self._detector: SubtitleSystem | None = None
        self._analyzing = False
        self._lock = threading.Lock()
        self._valid_tabs: list[tuple[bytes, float, str]] = []  # last 2 valid (png_bytes, timestamp, filename)
        self._hero_crops: dict[str, bytes] = {}  # hero_name -> cropped PNG bytes
        self._use_telegram = use_telegram
        self._use_mcp = use_mcp
        self._use_transcript = use_transcript
        self._replay_source = replay_source  # ReplaySource for replay mode
        self._no_analysis = no_analysis
        self._engine: CaptureEngine | None = None
        self._tick_loop: TickLoop | None = None
        self._subtitle_system: SubtitleSystem | None = None
        self._chat_system: ChatSystem | None = None
        self._bus = event_bus
        self._icon: pystray.Icon | None = None
        self._cooldown_until_tick: int = 0  # ignore tab/crop events until this tick
        self._poll_thread: threading.Thread | None = None
        self._poll_stop = threading.Event()
        self._recording = False
        if event_bus:
            self._register_commands(event_bus)

    def _register_commands(self, bus: EventBus) -> None:
        """Register command handlers on the event bus."""
        bus.register("start_listening", self._start_listening)
        bus.register("stop_listening", self._stop_listening)
        bus.register("toggle_recording", lambda: self._on_toggle_recording(None, None))  # type: ignore[arg-type]
        bus.register("submit_win", lambda: self._on_submit_tab("VICTORY"))
        bus.register("submit_loss", lambda: self._on_submit_tab("DEFEAT"))
        bus.register("quit", self._ws_quit)

    def _ws_quit(self) -> None:
        """Handle quit command from companion app."""
        _logger.info("Quit requested via WebSocket command")
        self._shutdown()

    def _in_post_detection_cooldown(self) -> bool:
        """Check if tab/crop events should be ignored (post-detection cooldown)."""
        if self._cooldown_until_tick <= 0:
            return False
        if self._tick_loop is None:
            return False
        return self._tick_loop._current_tick < self._cooldown_until_tick

    def _ws_emit(self, event: dict) -> None:
        """Emit an event to the WebSocket bus if enabled."""
        if self._bus:
            self._bus.emit(event)

    def store_valid_tab(self, png_bytes: bytes, timestamp: float, filename: str) -> None:
        """Store a valid Tab screenshot (called by TabCaptureSystem).

        timestamp is time.monotonic() in live mode, sim_time in replay mode.
        """
        if self._in_post_detection_cooldown():
            _logger.debug(f"Tab ignored (post-detection cooldown): {filename}")
            return
        with self._lock:
            self._valid_tabs.append((png_bytes, timestamp, filename))
            if len(self._valid_tabs) > 2:
                self._valid_tabs.pop(0)
            tab_count = len(self._valid_tabs)
        print_status(f"Tab screenshot saved: {filename}")
        self._ws_emit({"type": "tab_capture", "filename": filename,
                       "timestamp": timestamp, "count": tab_count})

    def store_hero_crop(self, name: str, crop: bytes) -> None:
        """Store a hero panel crop (called by TabCaptureSystem)."""
        if self._in_post_detection_cooldown():
            return
        with self._lock:
            if not any(_edit_distance(name.lower(), k.lower()) <= 2
                       for k in self._hero_crops):
                self._hero_crops[name] = crop
                _logger.info(f"Stored hero crop: {name}")
                self._ws_emit({"type": "hero_crop", "name": name})
            else:
                _logger.debug(f"Hero crop dedup skip: {name}")

    def _on_hero_switch(self, player: str, hero: str, sim_time: float) -> None:
        """Callback when a hero switch is detected in subtitles."""
        self._ws_emit({"type": "hero_switch", "player": player,
                       "hero": hero, "time": sim_time})

    def _on_player_change(self, player: str, event: str, sim_time: float) -> None:
        """Callback when a player joins or leaves the game."""
        self._ws_emit({"type": "player_change", "player": player,
                       "event": event, "time": sim_time})

    def _on_detected(self, result: str, detection_time: float = 0.0) -> None:
        """Immediate callback when VICTORY/DEFEAT is first detected."""
        self._ws_emit({"type": "detection", "result": result, "time": detection_time})
        show_notification("OverwatchLooker", f"{result} detected! Analyzing...")
        if self._no_analysis:
            print_status(f"Detected: {result} (analysis skipped)")
        else:
            print_status(f"Detected: {result}. Analyzing after delay...")

    def _on_detection(self, result: str, detection_time: float = 0.0) -> None:
        """Delayed callback: start analysis after the tick-based delay has elapsed."""
        if self._no_analysis:
            return
        # Start cooldown: ignore tab/crop events for 30s to prevent stale data leaking
        if self._tick_loop:
            fps = self._tick_loop.fps
            self._cooldown_until_tick = (
                self._tick_loop._current_tick + int(fps * _POST_SUBMIT_COOLDOWN)
            )
        with self._lock:
            if self._analyzing:
                print_status("Analysis already in progress, ignoring detection trigger.")
                return
            self._analyzing = True
            hero_crops = dict(self._hero_crops)
            self._hero_crops.clear()

        # Grab hero map and history from subtitle listener before resetting
        hero_map: dict[str, str] = {}
        hero_history: dict[str, list[tuple[float, str]]] = {}
        if self._detector is not None:
            hero_map = self._detector.hero_map
            hero_history = self._detector.hero_history
            self._detector.reset_match()

        player_changes: list[tuple[float, str, str]] = []
        if self._chat_system is not None:
            player_changes = self._chat_system.player_changes
            self._chat_system.reset_match()

        thread = threading.Thread(target=self._run_analysis,
                                  args=(result, detection_time, hero_map, hero_history, hero_crops, player_changes), daemon=True)
        thread.start()

    def _run_analysis(self, detection_result: str, detection_time: float = 0.0,
                      hero_map: dict[str, str] | None = None,
                      hero_history: dict[str, list[tuple[float, str]]] | None = None,
                      hero_crops: dict[str, bytes] | None = None,
                      player_changes: list[tuple[float, str, str]] | None = None) -> None:
        """Wait, then analyze last valid tab screenshot (fall back to previous if rejected)."""
        try:
            print_status(f"Analyzing {detection_result}...")
            self._ws_emit({"type": "analyzing", "result": detection_result})

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
                merge_heroes(result, hero_map=hero_map, hero_history=hero_history,
                             player_changes=player_changes)
                display_text = format_match(result, hero_map=hero_map,
                                            hero_history=hero_history)

                formatted = print_analysis(display_text)
                self._ws_emit({"type": "analysis", "data": result})
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

    # ------------------------------------------------------------------
    # Engine lifecycle: polling for Overwatch window
    # ------------------------------------------------------------------

    def _is_overwatch_running(self) -> bool:
        """Check if any window belongs to overwatch.exe."""
        import ctypes
        import ctypes.wintypes

        # EnumWindows approach: check all top-level windows for overwatch.exe
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        found = [False]

        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def enum_callback(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
            if not handle:
                return True
            try:
                buf = ctypes.create_unicode_buffer(260)
                size = ctypes.wintypes.DWORD(260)
                if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    exe = buf.value.rsplit("\\", 1)[-1].lower()
                    if exe == "overwatch.exe":
                        found[0] = True
                        return False  # stop enumeration
            finally:
                kernel32.CloseHandle(handle)
            return True

        user32.EnumWindows(enum_callback, 0)
        return found[0]

    def _poll_for_overwatch(self) -> None:
        """Polling thread: waits for OW window, creates engine, starts tick loop."""
        stable_since: float | None = None

        while not self._poll_stop.is_set():
            if self._engine is not None:
                # Engine exists — monitor health
                self._monitor_engine_health()
                self._poll_stop.wait(_OW_POLL_INTERVAL)
                continue

            ow_present = self._is_overwatch_running()

            if ow_present:
                if stable_since is None:
                    stable_since = time.monotonic()
                    _logger.info("Overwatch window detected, waiting for stability...")
                elif time.monotonic() - stable_since >= _OW_STABILITY_TIME:
                    # Stable long enough — create engine and start
                    try:
                        self._create_engine_and_start()
                        stable_since = None
                    except Exception as e:
                        _logger.error(f"Failed to create capture engine: {e}")
                        stable_since = None
            else:
                if stable_since is not None:
                    _logger.info("Overwatch window disappeared during stability wait")
                    stable_since = None

            self._poll_stop.wait(_OW_POLL_INTERVAL)

    def _create_engine_and_start(self) -> None:
        """Create a memoir CaptureEngine targeting Overwatch and start the tick loop."""
        from memoir_capture import CaptureEngine, MetaKeyEntry, WindowExeTarget

        fps = 10
        key_map = [
            MetaKeyEntry(bit_index=0, virtual_key=0x09, name="tab"),
            MetaKeyEntry(bit_index=1, virtual_key=0xA4, name="alt_l"),
            MetaKeyEntry(bit_index=2, virtual_key=0xA5, name="alt_r"),
        ]
        self._engine = CaptureEngine(
            WindowExeTarget("(?i)overwatch"),
            max_fps=fps,
            key_map=key_map,
        )
        self._engine.start()

        from overwatchlooker.tick import (
            ChatSystem, MemoirFrameSource, MemoirInputSource,
            SubtitleSystem, TabCaptureSystem, TickLoop,
        )
        from overwatchlooker.config import SUBTITLE_POLL_INTERVAL

        frame_source = MemoirFrameSource(self._engine, fps)
        input_source = MemoirInputSource(frame_source, [
            {"bit_index": k.bit_index, "name": k.name} for k in key_map
        ])

        self._tick_loop = TickLoop(fps, frame_source, input_source)

        tab_system = TabCaptureSystem(self, fps=fps)
        self._tick_loop.register(tab_system.on_tick, every_n_ticks=1)

        subtitle_interval = max(1, int(fps * SUBTITLE_POLL_INTERVAL))
        detection_delay = int(fps * _AUDIO_ANALYSIS_DELAY)
        subtitle_system = SubtitleSystem(on_match=self._on_detection,
                                         on_detected=self._on_detected,
                                         on_hero_switch=self._on_hero_switch,
                                         transcript=self._use_transcript,
                                         detection_delay_ticks=detection_delay)
        self._tick_loop.register(subtitle_system.on_tick, every_n_ticks=subtitle_interval)
        self._detector = subtitle_system
        self._subtitle_system = subtitle_system

        chat_system = ChatSystem(on_player_change=self._on_player_change)
        self._tick_loop.register(chat_system.on_tick, every_n_ticks=subtitle_interval)
        self._chat_system = chat_system

        tick_thread = threading.Thread(target=self._tick_loop.run, daemon=True)
        tick_thread.start()

        print_status("Overwatch detected — capture engine started.")
        self._ws_emit({"type": "state", "active": True, "analyzing": False})

    def _monitor_engine_health(self) -> None:
        """Check if the engine has faulted (e.g. OW closed)."""
        if self._engine is None:
            return
        err = self._engine.get_last_error()
        if err:
            _logger.warning(f"Capture engine error: {err}")
            self._tear_down_engine()
            print_status("Overwatch closed. Waiting for reconnect...")
            self._ws_emit({"type": "state", "active": True, "analyzing": False})
            if self._recording:
                self._recording = False
                self._ws_emit({"type": "state", "recording": False})

    def _tear_down_engine(self) -> None:
        """Stop tick loop and engine, but keep polling alive."""
        if self._tick_loop:
            self._tick_loop.stop()
            self._tick_loop = None
        if hasattr(self, '_subtitle_system') and self._subtitle_system:
            self._subtitle_system.close()
            self._subtitle_system = None
        self._chat_system = None
        self._detector = None
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

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

        if self._replay_source:
            # Replay mode: use ReplayFrameSource/ReplayInputSource directly
            from overwatchlooker.tick import (
                ChatSystem, ReplayFrameSource, ReplayInputSource,
                SubtitleSystem, TabCaptureSystem, TickLoop,
            )
            from overwatchlooker.config import SUBTITLE_POLL_INTERVAL

            fps = self._replay_source.fps
            frame_source = ReplayFrameSource(self._replay_source.reader)
            input_source = ReplayInputSource(self._replay_source.events)

            self._tick_loop = TickLoop(fps, frame_source, input_source)

            tab_system = TabCaptureSystem(self, fps=fps)
            self._tick_loop.register(tab_system.on_tick, every_n_ticks=1)

            subtitle_interval = max(1, int(fps * SUBTITLE_POLL_INTERVAL))
            detection_delay = int(fps * _AUDIO_ANALYSIS_DELAY)
            subtitle_system = SubtitleSystem(on_match=self._on_detection,
                                             on_detected=self._on_detected,
                                             on_hero_switch=self._on_hero_switch,
                                             transcript=self._use_transcript,
                                             detection_delay_ticks=detection_delay)
            self._tick_loop.register(subtitle_system.on_tick, every_n_ticks=subtitle_interval)
            self._detector = subtitle_system
            self._subtitle_system = subtitle_system

            chat_system = ChatSystem(on_player_change=self._on_player_change)
            self._tick_loop.register(chat_system.on_tick, every_n_ticks=subtitle_interval)
            self._chat_system = chat_system

            detect_mode = "replay"
        else:
            # Live mode: start polling thread for OW window
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_for_overwatch, daemon=True
            )
            self._poll_thread.start()
            detect_mode = "subtitle"

        print_status(f"Listening (analyzer={ANALYZER}, detection={detect_mode}). Tab=screenshot.")
        self._ws_emit({"type": "state", "active": True, "analyzing": False})

    def _stop_listening(self) -> None:
        if not self._active:
            return
        self._active = False

        # Stop polling thread
        self._poll_stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None

        # Stop engine and tick loop
        self._tear_down_engine()

        # Also stop tick loop if it was set up for replay
        if self._tick_loop:
            self._tick_loop.stop()
            self._tick_loop = None

        print_status("Stopped listening.")
        self._ws_emit({"type": "state", "active": False})

    def _on_submit_tab(self, result: str) -> None:
        """Manually submit last tab screenshot with a given result."""
        if self._tick_loop:
            fps = self._tick_loop.fps
            self._cooldown_until_tick = (
                self._tick_loop._current_tick + int(fps * _POST_SUBMIT_COOLDOWN)
            )
        with self._lock:
            if self._analyzing:
                print_status("Analysis already in progress.")
                return
            self._analyzing = True
            hero_crops = dict(self._hero_crops)
            self._hero_crops.clear()

        hero_map: dict[str, str] = {}
        hero_history: dict[str, list[tuple[float, str]]] = {}
        if self._detector is not None:
            hero_map = self._detector.hero_map
            hero_history = self._detector.hero_history
            self._detector.reset_match()

        player_changes: list[tuple[float, str, str]] = []
        if self._chat_system is not None:
            player_changes = self._chat_system.player_changes
            self._chat_system.reset_match()

        detection_time = self._tick_loop._current_tick / self._tick_loop.fps if self._tick_loop else 0.0
        thread = threading.Thread(target=self._run_analysis,
                                  args=(result, detection_time, hero_map, hero_history, hero_crops, player_changes), daemon=True)
        thread.start()

    def _on_submit_win(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_submit_tab("VICTORY")

    def _on_submit_loss(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_submit_tab("DEFEAT")

    def _on_toggle_recording(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Toggle recording on/off via the memoir engine."""
        if self._engine is None:
            print_error("No capture engine running. Start Overwatch first.")
            return

        if self._recording:
            try:
                self._engine.stop_recording()
                self._recording = False
                print_status("Recording stopped.")
                show_notification("OverwatchLooker", "Recording saved.")
                self._ws_emit({"type": "state", "recording": False})
            except Exception as e:
                print_error(f"Failed to stop recording: {e}")
        else:
            try:
                _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                rec_dir = _RECORDINGS_DIR / timestamp
                rec_dir.mkdir()
                base_path = rec_dir / "recording"
                info = self._engine.start_recording(str(base_path))
                self._recording = True
                print_status(f"Recording to {info.video_path}")
                show_notification("OverwatchLooker", "Recording started.")
                self._ws_emit({"type": "state", "recording": True})
            except Exception as e:
                print_error(f"Failed to start recording: {e}")
        self._rebuild_menu()

    def _on_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        if self._engine and self._recording:
            try:
                self._engine.stop_recording()
                self._recording = False
            except Exception:
                pass
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
        if self._icon is None:
            return
        label = "Stop Listening" if self._active else "Start Listening"
        rec_label = "Stop Recording" if self._recording else "Start Recording"
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
            _logger.info("KeyboardInterrupt received")
            self._shutdown()

        if not tray_thread.is_alive():
            _logger.warning("Tray icon thread exited unexpectedly")
            self._shutdown()

    def _shutdown(self) -> None:
        print_status("Shutting down...")
        if self._engine and self._recording:
            try:
                self._engine.stop_recording()
                self._recording = False
                print_status("Recording saved on shutdown.")
            except Exception:
                pass
        self._stop_listening()
        if self._icon:
            self._icon.stop()
