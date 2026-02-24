import ctypes
import ctypes.wintypes
import threading
from collections.abc import Callable

from pynput import keyboard

_user32 = ctypes.windll.user32
_psapi = ctypes.windll.psapi
_kernel32 = ctypes.windll.kernel32

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _get_foreground_exe() -> str:
    """Return the lowercase executable name of the foreground window."""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return ""
    pid = ctypes.wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = ctypes.wintypes.DWORD(260)
        if _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value.rsplit("\\", 1)[-1].lower()
        return ""
    finally:
        _kernel32.CloseHandle(handle)


class HotkeyListener:
    def __init__(self, on_tab_press: Callable[[], None],
                 on_tab_release: Callable[[], None] | None = None):
        self._on_tab_press = on_tab_press
        self._on_tab_release = on_tab_release
        self._tab_fired = False
        self._alt_held = False
        self._lock = threading.Lock()
        self._kb_listener: keyboard.Listener | None = None

    def start(self) -> None:
        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._kb_listener.start()

    def stop(self) -> None:
        if self._kb_listener:
            self._kb_listener.stop()

    def _on_key_press(self, key, *args) -> None:
        if key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
            with self._lock:
                self._alt_held = True
            return

        if key == keyboard.Key.tab:
            with self._lock:
                if self._tab_fired or self._alt_held:
                    return
                self._tab_fired = True
            if _get_foreground_exe() == "overwatch.exe":
                self._on_tab_press()

    def _on_key_release(self, key, *args) -> None:
        if key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
            with self._lock:
                self._alt_held = False
        elif key == keyboard.Key.tab:
            with self._lock:
                self._tab_fired = False
            if self._on_tab_release:
                self._on_tab_release()
