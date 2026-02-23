import logging
import threading
import time
import traceback

import pystray
from PIL import Image, ImageDraw, ImageFont

from overwatchlooker.audio_listener import AudioListener
from overwatchlooker.ocr_analyzer import analyze_screenshot
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


class App:
    def __init__(self):
        self._active = False
        self._hotkey: HotkeyListener | None = None
        self._audio: AudioListener | None = None
        self._analyzing = False
        self._lock = threading.Lock()
        self._last_valid_tab: bytes | None = None  # last screenshot that passed tab check

    def _on_tab_press(self) -> None:
        """Tab press: capture, validate, and save screenshot (no analysis)."""
        def _delayed_capture():
            try:
                time.sleep(1.0)
                png_bytes = capture_monitor()
                saved_path = save_screenshot(png_bytes)
                if is_ow2_tab_screen(png_bytes):
                    self._last_valid_tab = png_bytes
                    print_status(f"Tab screenshot saved to {saved_path}")
                else:
                    print_status(f"Screenshot saved (not a Tab screen): {saved_path}")
            except Exception as e:
                print_error(f"Screenshot capture failed: {e}")
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

            png_bytes = self._last_valid_tab
            if png_bytes is None:
                print_error("No valid Tab screenshot found. Press Tab during the scoreboard first.")
                show_notification("OverwatchLooker", "No Tab screenshot to analyze.")
                return

            print_status(f"Analyzing last valid Tab screenshot ({len(png_bytes)} bytes)...")
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
        self._active = True
        self._hotkey = HotkeyListener(on_tab_press=self._on_tab_press)
        self._hotkey.start()
        self._audio = AudioListener(on_match=self._on_audio_match)
        self._audio.start()
        print_status("Listening for Tab (screenshot) and audio (VICTORY/DEFEAT)...")

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
