"""TFLite inference wrapper. Same code on desktop now and on the Raspberry Pi later.

Backend preference: ai_edge_litert -> tflite_runtime -> tensorflow.lite. On the Pi
only the small runtime package is needed, not full TensorFlow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai_edge_litert"
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite_runtime"
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow"
    except ImportError as exc:
        raise ImportError(
            "No TFLite interpreter found. Install ai-edge-litert, tflite-runtime "
            "or tensorflow."
        ) from exc


class TFLiteAutoencoder:
    """Runs a (1, W, C) -> (1, W, C) autoencoder .tflite, float32 or full-int8 I/O."""

    def __init__(self, model_path: str | Path | None = None,
                 model_content: bytes | None = None, num_threads: int = 1) -> None:
        if (model_path is None) == (model_content is None):
            raise ValueError("give exactly one of model_path or model_content")
        cls, self.backend = _interpreter_class()
        kwargs: dict = {"num_threads": num_threads}
        if model_path is not None:
            kwargs["model_path"] = str(model_path)
        else:
            kwargs["model_content"] = bytes(model_content)
        self._it = cls(**kwargs)
        self._it.allocate_tensors()
        inp = self._it.get_input_details()[0]
        out = self._it.get_output_details()[0]
        self._in, self._out = inp, out
        shape = tuple(int(d) for d in inp["shape"])
        if len(shape) != 3 or shape[0] != 1:
            raise ValueError(f"expected input shape (1, W, C), got {shape}")
        self.input_shape = shape
        self.window_size, self.n_channels = shape[1], shape[2]
        self.is_quantized = bool(np.issubdtype(inp["dtype"], np.integer))

    # -- quantization helpers -------------------------------------------------
    def _quantize_unclipped(self, x: np.ndarray) -> np.ndarray:
        scale, zp = self._in["quantization"]
        return np.round(x / scale) + zp

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        info = np.iinfo(self._in["dtype"])
        return np.clip(self._quantize_unclipped(x), info.min, info.max).astype(self._in["dtype"])

    def _check(self, windows: np.ndarray) -> np.ndarray:
        w = np.asarray(windows, dtype=np.float32)
        if w.ndim != 3 or w.shape[1:] != self.input_shape[1:]:
            raise ValueError(f"expected (N, {self.window_size}, {self.n_channels}), got {w.shape}")
        return w

    def input_clip_fraction(self, windows: np.ndarray) -> float:
        """Fraction of input values outside the representable int8 range (0 for float)."""
        w = self._check(windows)
        if not self.is_quantized or w.size == 0:
            return 0.0
        info = np.iinfo(self._in["dtype"])
        q = self._quantize_unclipped(w)
        return float(((q < info.min) | (q > info.max)).mean())

    # -- inference --------------------------------------------------------------
    def reconstruct(self, windows: np.ndarray) -> np.ndarray:
        """Reconstruct windows one at a time (batch 1, as deployed). Returns float32."""
        w = self._check(windows)
        out = np.empty_like(w)
        feed = self._quantize(w) if self.is_quantized else w
        o_scale, o_zp = self._out["quantization"]
        for i in range(w.shape[0]):
            self._it.set_tensor(self._in["index"], feed[i : i + 1])
            self._it.invoke()
            y = self._it.get_tensor(self._out["index"])
            if np.issubdtype(self._out["dtype"], np.integer):
                y = (y.astype(np.float32) - o_zp) * o_scale
            out[i] = y[0]
        return out

    def reconstruction_errors(self, windows: np.ndarray) -> np.ndarray:
        """Per-window MSE against the ORIGINAL float window, shape (N,)."""
        w = self._check(windows)
        if w.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        return ((self.reconstruct(w) - w) ** 2).mean(axis=(1, 2)).astype(np.float64)

    def describe(self) -> dict:
        def q(d):
            scale, zp = d["quantization"]
            return {"dtype": np.dtype(d["dtype"]).name, "scale": float(scale), "zero_point": int(zp)}
        return {"backend": self.backend, "input_shape": list(self.input_shape),
                "input": q(self._in), "output": q(self._out)}
