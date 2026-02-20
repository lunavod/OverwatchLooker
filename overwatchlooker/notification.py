import io
import math
import struct
import subprocess
import threading
import tkinter as tk
import wave
import winsound

import mss

NOTIFICATION_VOLUME = 0.20  # 20% amplitude


def copy_to_clipboard(text: str) -> None:
    """Copy text to the system clipboard via clip.exe."""
    subprocess.run("clip", input=text.encode("utf-16-le"), check=True)


def _get_second_monitor_geometry() -> dict:
    """Get the geometry of monitor 2. Falls back to monitor 1 if only one exists."""
    with mss.mss() as sct:
        if len(sct.monitors) > 2:
            return sct.monitors[2]  # index 2 = second physical monitor
        return sct.monitors[1]  # fallback to primary


def _play_notification_sound() -> None:
    """Generate and play a gentle two-tone chime at 20% volume."""
    sample_rate = 44100

    # Two-note chime: C5 -> E5 (major third, sounds pleasant)
    notes = [
        (523.25, 0.12),  # C5 for 120ms
        (659.25, 0.18),  # E5 for 180ms
    ]

    frames = []
    for freq, duration in notes:
        n_samples = int(sample_rate * duration)
        for i in range(n_samples):
            t = i / sample_rate
            # Fade in/out envelope to avoid clicks
            envelope = 1.0
            fade = 0.02  # 20ms fade
            if t < fade:
                envelope = t / fade
            elif t > duration - fade:
                envelope = (duration - t) / fade
            sample = NOTIFICATION_VOLUME * envelope * math.sin(2 * math.pi * freq * t)
            frames.append(struct.pack("<h", int(sample * 32767)))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)


def show_notification(title: str, message: str, duration_ms: int = 4000) -> None:
    """Show a small notification window on the second monitor."""
    thread = threading.Thread(
        target=_show_notification_window,
        args=(title, message, duration_ms),
        daemon=True,
    )
    thread.start()


def _show_notification_window(title: str, message: str, duration_ms: int) -> None:
    mon = _get_second_monitor_geometry()

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#1a1a2e")

    # Size and position: bottom-right of second monitor
    width = 360
    height = 80
    x = mon["left"] + mon["width"] - width - 20
    y = mon["top"] + mon["height"] - height - 120
    root.geometry(f"{width}x{height}+{x}+{y}")

    title_label = tk.Label(
        root,
        text=title,
        font=("Arial", 11, "bold"),
        fg="#3399ff",
        bg="#1a1a2e",
        anchor="w",
    )
    title_label.pack(fill="x", padx=10, pady=(8, 0))

    msg_label = tk.Label(
        root,
        text=message,
        font=("Arial", 9),
        fg="#cccccc",
        bg="#1a1a2e",
        anchor="w",
        wraplength=340,
    )
    msg_label.pack(fill="x", padx=10, pady=(2, 8))

    threading.Thread(target=_play_notification_sound, daemon=True).start()

    root.after(duration_ms, root.destroy)
    root.mainloop()
