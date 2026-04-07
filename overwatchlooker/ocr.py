"""ONNX Runtime OCR inference wrapper.

Drop-in replacement for PaddleX's ``create_model().predict()`` interface.
Loads an ONNX model exported from PP-OCRv5 server rec and provides the
same ``predict(img_bgr)`` → ``[{"rec_text": str, "rec_score": float}]``
API used throughout the codebase.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

_logger = logging.getLogger("overwatchlooker")

# Target input shape (matches PP-OCRv5 server rec training config)
_IMG_C, _IMG_H, _IMG_W = 3, 48, 320


class OnnxRecModel:
    """ONNX Runtime text recognition model."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        onnx_path = model_dir / "inference.onnx"
        dict_path = model_dir / "dict.txt"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"])

        # Build character list: blank (index 0) + dict chars + trailing space
        self._charset: list[str] = ["blank"]
        with open(dict_path, encoding="utf-8") as f:
            for line in f:
                self._charset.append(line.strip())
        self._charset.append(" ")

    def predict(self, img_bgr: np.ndarray) -> Iterator[dict[str, str | float]]:
        """Run OCR on a single BGR image. Yields one result dict."""
        inp = self._preprocess(img_bgr)
        logits = self._sess.run(None, {"x": inp})[0]  # (1, T, C)
        text, score = self._ctc_decode(logits[0])
        yield {"rec_text": text, "rec_score": score}

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize, normalize, pad to model input shape. Returns (1, C, H, W)."""
        h, w = img.shape[:2]
        ratio = w / float(h)
        max_wh = max(_IMG_W / _IMG_H, ratio)
        rw = min(int(math.ceil(_IMG_H * ratio)), int(_IMG_H * max_wh))
        resized = cv2.resize(img, (rw, _IMG_H)).astype(np.float32)
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5
        pad_w = int(_IMG_H * max_wh)
        padded = np.zeros((_IMG_C, _IMG_H, pad_w), dtype=np.float32)
        padded[:, :, :rw] = resized
        return padded[np.newaxis]

    def _ctc_decode(self, preds: np.ndarray) -> tuple[str, float]:
        """CTC greedy decode: argmax → collapse repeats → strip blanks."""
        idx = preds.argmax(axis=-1)
        prob = preds.max(axis=-1)
        # Remove consecutive duplicates
        mask = np.ones(len(idx), dtype=bool)
        mask[1:] = idx[1:] != idx[:-1]
        # Remove blank token (index 0)
        mask &= idx != 0
        chars = [self._charset[i] for i in idx[mask]]
        conf = prob[mask]
        text = "".join(chars)
        score = float(np.mean(conf)) if len(conf) > 0 else 0.0
        return text, score
