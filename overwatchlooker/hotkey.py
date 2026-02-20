import threading
from collections.abc import Callable

from pynput import keyboard, mouse


class HotkeyListener:
    def __init__(self, on_trigger: Callable[[], None]):
        self._on_trigger = on_trigger
        self._tab_held = False
        self._lock = threading.Lock()
        self._kb_listener: keyboard.Listener | None = None
        self._mouse_listener: mouse.Listener | None = None

    def start(self) -> None:
        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._mouse_listener = mouse.Listener(
            on_click=self._on_click,
        )
        self._kb_listener.start()
        self._mouse_listener.start()

    def stop(self) -> None:
        if self._kb_listener:
            self._kb_listener.stop()
        if self._mouse_listener:
            self._mouse_listener.stop()

    def _on_key_press(self, key, *args) -> None:
        if key == keyboard.Key.tab:
            with self._lock:
                self._tab_held = True

    def _on_key_release(self, key, *args) -> None:
        if key == keyboard.Key.tab:
            with self._lock:
                self._tab_held = False

    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool, *args) -> None:
        if pressed and button in (mouse.Button.x1, mouse.Button.x2):
            with self._lock:
                if self._tab_held:
                    self._on_trigger()
