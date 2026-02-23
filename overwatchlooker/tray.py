import threading

import pystray
from PIL import Image, ImageDraw, ImageFont

from overwatchlooker.ocr_analyzer import analyze_screenshot
from overwatchlooker.display import print_analysis, print_error, print_status
from overwatchlooker.hotkey import HotkeyListener
from overwatchlooker.notification import copy_to_clipboard, show_notification
from overwatchlooker.screenshot import capture_monitor, save_screenshot


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


class App:
    def __init__(self):
        self._active = False
        self._hotkey: HotkeyListener | None = None
        self._analyzing = False
        self._lock = threading.Lock()

    def _on_hotkey_trigger(self) -> None:
        """Called from pynput listener thread when hotkey fires."""
        with self._lock:
            if self._analyzing:
                print_status("Analysis already in progress, ignoring trigger.")
                return
            self._analyzing = True

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self) -> None:
        """Capture screenshot, run OCR analysis, print result."""
        try:
            print_status("Hotkey detected! Capturing screenshot...")
            png_bytes = capture_monitor()
            saved_path = save_screenshot(png_bytes)
            print_status(f"Screenshot saved to {saved_path}")
            print_status(f"Screenshot captured ({len(png_bytes)} bytes). Analyzing with OCR...")
            result = analyze_screenshot(png_bytes)
            if result.startswith("NOT_OW2_TAB"):
                print_error("Screenshot does not appear to be an OW2 Tab screen.")
                show_notification("OverwatchLooker", "Not an OW2 Tab screen.")
            else:
                formatted = print_analysis(result)
                copy_to_clipboard(formatted)
                show_notification("OverwatchLooker", "Analysis complete. Copied to clipboard.")
        except Exception as e:
            print_error(f"Analysis failed: {e}")
        finally:
            with self._lock:
                self._analyzing = False

    def _start_listening(self) -> None:
        if self._active:
            return
        self._active = True
        self._hotkey = HotkeyListener(on_trigger=self._on_hotkey_trigger)
        self._hotkey.start()
        print_status("Listening for hotkey (Tab + Mouse Side Button)...")

    def _stop_listening(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._hotkey:
            self._hotkey.stop()
            self._hotkey = None
        print_status("Stopped listening.")

    def _on_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._stop_listening()
        icon.stop()

    def _on_toggle(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        if self._active:
            self._stop_listening()
        else:
            self._start_listening()
        icon.update_menu()

    def run(self) -> None:
        """Main entry point -- blocks on the tray icon run loop."""
        self._icon = pystray.Icon(
            name="OverwatchLooker",
            icon=_create_icon_image(),
            title="OverwatchLooker",
            menu=pystray.Menu(
                pystray.MenuItem(
                    lambda item: "Stop Listening" if self._active else "Start Listening",
                    self._on_toggle,
                    default=True,
                ),
                pystray.MenuItem("Quit", self._on_quit),
            ),
        )

        def setup(icon: pystray.Icon):
            icon.visible = True
            self._start_listening()

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
