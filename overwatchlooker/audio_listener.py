"""Audio-based VICTORY/DEFEAT detection via per-process WASAPI loopback."""

import ctypes
import ctypes.wintypes
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from overwatchlooker.config import (
    AUDIO_CHUNK_DURATION,
    AUDIO_CONFIRM_HOPS,
    AUDIO_COOLDOWN_SECONDS,
    AUDIO_HOP_DURATION,
    AUDIO_MATCH_MARGIN,
    AUDIO_MATCH_THRESHOLD,
    AUDIO_MIN_RMS,
)

_logger = logging.getLogger("overwatchlooker")

REFS_DIR = Path(__file__).parent.parent / "refs"

_OW_EXE = "overwatch.exe"
_PID_POLL_INTERVAL = 2.0  # seconds between PID discovery attempts

# ProcTap fixed output format on Windows
_SAMPLE_RATE = 48000
_CHANNELS = 2


# ---------------------------------------------------------------------------
# Process discovery via Windows API (no extra dependency)
# ---------------------------------------------------------------------------

_TH32CS_SNAPPROCESS = 0x00000002


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.wintypes.DWORD),
        ("cntUsage", ctypes.wintypes.DWORD),
        ("th32ProcessID", ctypes.wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", ctypes.wintypes.DWORD),
        ("cntThreads", ctypes.wintypes.DWORD),
        ("th32ParentProcessID", ctypes.wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _find_process_pid(exe_name: str) -> int | None:
    """Find PID of a running process by executable name."""
    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snapshot == -1:
        return None
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            return None
        while True:
            if entry.szExeFile.lower() == exe_name.lower():
                return entry.th32ProcessID
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _load_audio(path: Path, target_sr: int) -> np.ndarray | None:
    """Load an audio file (wav, ogg, flac, etc.) and return mono float32 samples."""
    try:
        samples, sr = sf.read(path, dtype='float32', always_2d=True)
    except Exception as e:
        _logger.error(f"Failed to load {path}: {e}")
        return None

    # Mix to mono
    samples = samples.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        duration = len(samples) / sr
        n_target = int(duration * target_sr)
        indices = np.linspace(0, len(samples) - 1, n_target)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
        _logger.info(f"Resampled {path.name} from {sr}Hz to {target_sr}Hz")

    return samples


def _ncc_1d(chunk: np.ndarray, ref: np.ndarray) -> float:
    """Time-domain normalized cross-correlation.

    Slides the reference over the chunk and returns the peak NCC (range [-1, 1]).
    Matches the exact waveform pattern, not just spectral shape.
    """
    if len(chunk) < len(ref):
        return 0.0

    n = len(ref)
    ref_mean = float(ref.mean())
    ref_std = float(ref.std())
    if ref_std < 1e-8:
        return 0.0

    ref_centered = ref - ref_mean
    corr = fftconvolve(chunk, ref_centered[::-1], mode='valid')

    # Sliding window standard deviation
    ones = np.ones(n)
    chunk_sum = fftconvolve(chunk, ones, mode='valid')
    chunk_sq_sum = fftconvolve(chunk ** 2, ones, mode='valid')
    chunk_var = chunk_sq_sum / n - (chunk_sum / n) ** 2
    chunk_std = np.sqrt(np.maximum(chunk_var, 1e-10))

    ncc = corr / (chunk_std * ref_std * n)
    return float(np.max(ncc))


class AudioListener:
    """Monitors Overwatch process audio for VICTORY/DEFEAT announcements."""

    def __init__(self, on_match: Callable[[str], None]):
        self._on_match = on_match
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_trigger_time = 0.0
        self._refs: dict[str, np.ndarray] = {}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        try:
            from proctap import ProcessAudioCapture
        except ImportError:
            _logger.error("proc-tap not installed. Audio listener disabled.")
            return

        # Load reference clips once
        self._load_references(_SAMPLE_RATE)
        if not self._refs:
            _logger.warning("No reference clips loaded. Audio listener will not detect anything.")
            return

        reconnect_delay = _PID_POLL_INTERVAL
        while self._running:
            # Wait for Overwatch process
            pid = self._wait_for_process()
            if pid is None:
                return  # _running went False

            # Re-verify PID still belongs to overwatch.exe (guards against PID reuse)
            verified = _find_process_pid(_OW_EXE)
            if verified != pid:
                _logger.warning(f"PID {pid} no longer belongs to {_OW_EXE}, retrying...")
                time.sleep(reconnect_delay)
                continue

            _logger.info(f"Audio: capturing from overwatch.exe (PID={pid})")
            capture = None
            try:
                capture = ProcessAudioCapture(pid=pid)
                capture.start()
                self._run_loop(capture)
                reconnect_delay = _PID_POLL_INTERVAL  # reset on successful run
            except Exception as e:
                _logger.warning(f"Audio capture error: {e}")
                reconnect_delay = min(reconnect_delay * 2, 30.0)  # backoff, cap 30s
            finally:
                if capture is not None:
                    try:
                        capture.close()
                    except Exception:
                        pass

            if self._running:
                _logger.info(f"Overwatch process lost. Reconnecting in {reconnect_delay:.0f}s...")
                time.sleep(reconnect_delay)

    def _wait_for_process(self) -> int | None:
        """Poll until overwatch.exe is found or listener is stopped."""
        _logger.info(f"Waiting for {_OW_EXE}...")
        while self._running:
            pid = _find_process_pid(_OW_EXE)
            if pid is not None:
                _logger.info(f"Found {_OW_EXE} (PID={pid})")
                return pid
            time.sleep(_PID_POLL_INTERVAL)
        return None

    def _run_loop(self, capture) -> None:
        sr = _SAMPLE_RATE
        channels = _CHANNELS

        hop_samples = int(sr * AUDIO_HOP_DURATION)  # mono samples per hop
        buffer_samples = int(sr * AUDIO_CHUNK_DURATION)
        ring_buffer = np.zeros(buffer_samples, dtype=np.float32)
        # Accumulator for small chunks from ProcTap
        pending: list[np.ndarray] = []
        pending_len = 0

        confirm_label: str | None = None
        confirm_count = 0
        last_loud_time = time.monotonic()
        silence_reconnect_threshold = 30.0  # reconnect if silent for this long

        _logger.info("Audio listener started.")

        while self._running:
            try:
                data = capture.read(timeout=1.0)
                if not data:
                    continue
            except RuntimeError:
                break  # capture stopped, trigger reconnect
            except Exception as e:
                _logger.warning(f"Audio stream read error: {e}")
                break

            # Convert to mono float32
            samples = np.frombuffer(data, dtype=np.float32)
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)

            pending.append(samples)
            pending_len += len(samples)

            # Wait until we've accumulated a full hop before processing
            if pending_len < hop_samples:
                continue

            # Merge accumulated chunks
            merged = np.concatenate(pending)
            pending.clear()
            pending_len = 0

            # Shift ring buffer and append new data (clamp to buffer size)
            if len(merged) > buffer_samples:
                merged = merged[-buffer_samples:]
            n = len(merged)
            ring_buffer = np.roll(ring_buffer, -n)
            ring_buffer[-n:] = merged

            # Check cooldown
            now = time.monotonic()
            if now - self._last_trigger_time < AUDIO_COOLDOWN_SECONDS:
                continue

            # Energy gate — skip matching when audio is too quiet
            rms = float(np.sqrt(np.mean(ring_buffer ** 2)))
            if rms < AUDIO_MIN_RMS:
                if now - last_loud_time > silence_reconnect_threshold:
                    _logger.info("Audio silent for 30s, reconnecting capture...")
                    break  # trigger reconnect in _run()
                continue
            last_loud_time = now

            # Match against references (1D time-domain NCC)
            scores: dict[str, float] = {}
            for label, ref_samples in self._refs.items():
                scores[label] = _ncc_1d(ring_buffer, ref_samples)
                _logger.debug(f"Audio NCC [{label}]: {scores[label]:.4f} (rms={rms:.4f})")

            if not scores:
                continue

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_label, best_score = ranked[0]

            # Must exceed threshold
            if best_score < AUDIO_MATCH_THRESHOLD:
                confirm_label = None
                confirm_count = 0
                continue

            # Must beat runner-up by margin
            if len(ranked) > 1:
                runner_up_score = ranked[1][1]
                if best_score - runner_up_score < AUDIO_MATCH_MARGIN:
                    _logger.debug(f"Audio match rejected: margin too small "
                                  f"({best_label}={best_score:.4f} vs "
                                  f"{ranked[1][0]}={runner_up_score:.4f})")
                    confirm_label = None
                    confirm_count = 0
                    continue

            # Require consecutive hops above threshold to confirm
            if best_label == confirm_label:
                confirm_count += 1
            else:
                confirm_label = best_label
                confirm_count = 1

            if confirm_count < AUDIO_CONFIRM_HOPS:
                _logger.debug(f"Audio confirm {best_label}: {confirm_count}/{AUDIO_CONFIRM_HOPS} "
                              f"(score={best_score:.4f})")
                continue

            _logger.info(f"Audio match: {best_label} (score={best_score:.4f}, "
                         f"confirmed over {confirm_count} hops)")
            self._last_trigger_time = now
            confirm_label = None
            confirm_count = 0
            self._on_match(best_label)

    def _load_references(self, target_sr: int) -> None:
        """Load reference audio clips."""
        SUPPORTED_EXTS = (".wav", ".ogg", ".flac")
        ref_names = {"VICTORY": "victory", "DEFEAT": "defeat"}
        for label, stem in ref_names.items():
            path = None
            for ext in SUPPORTED_EXTS:
                candidate = REFS_DIR / f"{stem}{ext}"
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                _logger.warning(f"Reference clip not found: {REFS_DIR / stem}.{{wav,ogg,flac}}")
                continue
            samples = _load_audio(path, target_sr)
            if samples is None:
                continue
            self._refs[label] = samples
            _logger.info(f"Loaded reference '{label}' from {path.name} "
                         f"({len(samples)/target_sr:.2f}s)")
