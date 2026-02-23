import logging
import threading
import time
import traceback

import pystray
from PIL import Image, ImageDraw, ImageFont

from overwatchlooker.audio_listener import AudioListener
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
    def __init__(self):
        self._active = False
        self._hotkey: HotkeyListener | None = None
        self._audio: AudioListener | None = None
        self._analyzing = False
        self._lock = threading.Lock()
        self._last_valid_tab: tuple[bytes, float] | None = None  # (png_bytes, timestamp)
        self._tab_pending = False  # debounce flag

    def _on_tab_press(self) -> None:
        """Tab press: capture, validate, and save screenshot (no analysis)."""
        with self._lock:
            if self._tab_pending:
                return
            self._tab_pending = True

        def _delayed_capture():
            try:
                time.sleep(1.0)
                png_bytes = capture_monitor()
                saved_path = save_screenshot(png_bytes)
                if is_ow2_tab_screen(png_bytes):
                    with self._lock:
                        self._last_valid_tab = (png_bytes, time.monotonic())
                    print_status(f"Tab screenshot saved to {saved_path}")
                else:
                    print_status(f"Screenshot saved (not a Tab screen): {saved_path}")
            except Exception as e:
                print_error(f"Screenshot capture failed: {e}")
            finally:
                # Release debounce after capture + small cooldown
                time.sleep(_TAB_DEBOUNCE - 1.0)
                with self._lock:
                    self._tab_pending = False

        threading.Thread(target=_delayed_capture, daemon=True).start()

    def _on_audio_match(self, result: str) -> None:
        """Audio detected VICTORY/DEFEAT: analyze the latest screenshot."""
        with self._lock:
            if self._analyzing:
                print_status("Analysis already in progress, ignoring audio trigger.")
                return
            self._analyzing = True

        thread = threading.Thread(target=self._run_analysis, args=(result,), daemon=True)
        thread.start()

    def _run_analysis(self, audio_result: str) -> None:
        """Wait, then analyze last valid tab screenshot."""
        try:
            print_status(f"Audio detected: {audio_result}. "
                         f"Waiting {_AUDIO_ANALYSIS_DELAY:.0f}s before analysis...")
            time.sleep(_AUDIO_ANALYSIS_DELAY)

            with self._lock:
                tab_data = self._last_valid_tab

            if tab_data is None:
                print_error("No valid Tab screenshot found. Press Tab during the scoreboard first.")
                show_notification("OverwatchLooker", "No Tab screenshot to analyze.")
                return

            png_bytes, tab_time = tab_data
            age = time.monotonic() - tab_time
            if age > SCREENSHOT_MAX_AGE_SECONDS:
                print_error(f"Last Tab screenshot is {age:.0f}s old (max {SCREENSHOT_MAX_AGE_SECONDS:.0f}s). "
                            "Press Tab during the scoreboard first.")
                show_notification("OverwatchLooker", "Tab screenshot too old.")
                return

            print_status(f"Analyzing Tab screenshot ({len(png_bytes)} bytes, "
                         f"{age:.0f}s old) with {ANALYZER} backend...")
            if ANALYZER == "claude":
                from overwatchlooker.analyzer import analyze_screenshot
            else:
                from overwatchlooker.ocr_analyzer import analyze_screenshot
            result = analyze_screenshot(png_bytes, audio_result=audio_result)
            if result.startswith("NOT_OW2_TAB"):
                print_error("Screenshot does not appear to be an OW2 Tab screen.")
                show_notification("OverwatchLooker", "Not an OW2 Tab screen.")
            else:
                formatted = print_analysis(result)
                copy_to_clipboard(formatted)
                show_notification("OverwatchLooker", f"{audio_result} — Analysis copied to clipboard.")
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
        self._hotkey = HotkeyListener(on_tab_press=self._on_tab_press)
        self._hotkey.start()
        self._audio = AudioListener(on_match=self._on_audio_match)
        self._audio.start()
        print_status(f"Listening (analyzer={ANALYZER}). Tab=screenshot, audio=VICTORY/DEFEAT.")

    def _stop_listening(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._hotkey:
            self._hotkey.stop()
            self._hotkey = None
        if self._audio:
            self._audio.stop()
            self._audio = None
        print_status("Stopped listening.")

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
