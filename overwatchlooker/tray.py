import logging
import threading
import time
import traceback

import pystray
from PIL import Image, ImageDraw, ImageFont

from overwatchlooker.config import ANALYZER, ANTHROPIC_API_KEY, SCREENSHOT_MAX_AGE_SECONDS
from overwatchlooker.display import print_analysis, print_error, print_status

_logger = logging.getLogger("overwatchlooker")
from overwatchlooker.hotkey import HotkeyListener
from overwatchlooker.notification import copy_to_clipboard, show_notification
from overwatchlooker.screenshot import (
    capture_monitor,
    is_ow2_tab_screen,
    save_screenshot,
)


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
    def __init__(self, use_telegram: bool = False, use_audio: bool = False):
        self._active = False
        self._hotkey: HotkeyListener | None = None
        self._detector = None  # SubtitleListener or AudioListener
        self._analyzing = False
        self._lock = threading.Lock()
        self._valid_tabs: list[tuple[bytes, float, str]] = []  # last 2 valid (png_bytes, timestamp, filename)
        self._tab_pending = False  # debounce flag
        self._tab_held = False
        self._use_telegram = use_telegram
        self._use_audio = use_audio

    def _on_tab_press(self) -> None:
        """Tab press: capture screenshots while held, retrying until valid."""
        with self._lock:
            if self._tab_pending:
                return
            self._tab_pending = True
            self._tab_held = True

        def _capture_loop():
            try:
                time.sleep(1.0)
                got_valid = False
                while True:
                    with self._lock:
                        if not self._tab_held:
                            break
                    png_bytes = capture_monitor()
                    saved_path = save_screenshot(png_bytes)
                    if is_ow2_tab_screen(png_bytes):
                        with self._lock:
                            self._valid_tabs.append((png_bytes, time.monotonic(), saved_path.name))
                            if len(self._valid_tabs) > 2:
                                self._valid_tabs.pop(0)
                        print_status(f"Tab screenshot saved to {saved_path}")
                        got_valid = True
                        break
                    else:
                        print_status(f"Screenshot saved (not a Tab screen): {saved_path}")
                    time.sleep(0.5)
                if not got_valid:
                    # Tab released before we got a valid screenshot — save last capture anyway
                    print_status("Tab released without valid Tab screen capture.")
            except Exception as e:
                print_error(f"Screenshot capture failed: {e}")
            finally:
                time.sleep(0.5)
                with self._lock:
                    self._tab_pending = False

        threading.Thread(target=_capture_loop, daemon=True).start()

    def _on_tab_release(self) -> None:
        with self._lock:
            self._tab_held = False

    def _on_detection(self, result: str) -> None:
        """Subtitle or audio detected VICTORY/DEFEAT: analyze the latest screenshot."""
        with self._lock:
            if self._analyzing:
                print_status("Analysis already in progress, ignoring detection trigger.")
                return
            self._analyzing = True

        thread = threading.Thread(target=self._run_analysis, args=(result,), daemon=True)
        thread.start()

    def _run_analysis(self, detection_result: str) -> None:
        """Wait, then analyze last valid tab screenshot (fall back to previous if rejected)."""
        try:
            print_status(f"Detected: {detection_result}. "
                         f"Waiting {_AUDIO_ANALYSIS_DELAY:.0f}s before analysis...")
            show_notification("OverwatchLooker", f"{detection_result} detected! Analyzing...")
            time.sleep(_AUDIO_ANALYSIS_DELAY)

            with self._lock:
                tabs = list(self._valid_tabs)

            if not tabs:
                print_error("No valid Tab screenshot found. Press Tab during the scoreboard first.")
                show_notification("OverwatchLooker", "No Tab screenshot to analyze.")
                return

            # Try most recent first, then fall back to previous
            for i, (png_bytes, tab_time, filename) in enumerate(reversed(tabs)):
                age = time.monotonic() - tab_time
                if age > SCREENSHOT_MAX_AGE_SECONDS:
                    continue

                is_fallback = i > 0
                if is_fallback:
                    time_diff = tabs[-1][1] - tab_time
                    print_status(f"Latest screenshot rejected by analyzer. "
                                 f"Falling back to previous ({time_diff:.0f}s older).")

                print_status(f"Analyzing {filename} ({len(png_bytes)} bytes, "
                             f"{age:.0f}s old) with {ANALYZER} backend...")
                if ANALYZER == "claude":
                    from overwatchlooker.analyzer import analyze_screenshot
                else:
                    from overwatchlooker.ocr_analyzer import analyze_screenshot
                result = analyze_screenshot(png_bytes, audio_result=detection_result)

                if result.startswith("NOT_OW2_TAB"):
                    print_status("Analyzer rejected screenshot as not OW2 Tab.")
                    continue

                formatted = print_analysis(result)
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

        if ANALYZER == "claude" and not ANTHROPIC_API_KEY:
            print_error("ANALYZER=claude but ANTHROPIC_API_KEY is not set. "
                        "Set it in .env or use ANALYZER=ocr.")
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
        self._hotkey = HotkeyListener(on_tab_press=self._on_tab_press,
                                      on_tab_release=self._on_tab_release)
        self._hotkey.start()
        if self._use_audio:
            from overwatchlooker.audio_listener import AudioListener
            self._detector = AudioListener(on_match=self._on_detection)
            detect_mode = "audio"
        else:
            from overwatchlooker.subtitle_listener import SubtitleListener
            self._detector = SubtitleListener(on_match=self._on_detection)
            detect_mode = "subtitle"
        self._detector.start()
        print_status(f"Listening (analyzer={ANALYZER}, detection={detect_mode}). Tab=screenshot.")

    def _stop_listening(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._hotkey:
            self._hotkey.stop()
            self._hotkey = None
        if self._detector:
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

        thread = threading.Thread(target=self._run_analysis, args=(result,), daemon=True)
        thread.start()

    def _on_submit_win(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_submit_tab("VICTORY")

    def _on_submit_loss(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_submit_tab("DEFEAT")

    def _on_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
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
        self._icon.menu = pystray.Menu(
            pystray.MenuItem(label, self._on_toggle, default=True),
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
        self._stop_listening()
        if self._icon:
            self._icon.stop()
