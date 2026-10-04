"""TFLite inference wrapper. Same code on desktop and on the Raspberry Pi.

Backend preference: ai_edge_litert -> tflite_runtime -> tensorflow.lite. On the Pi
only the small runtime package is needed, not full TensorFlow. numpy-only otherwise.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from time import perf_counter_ns

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


def _no_default_delegates(cls):
    enum = getattr(importlib.import_module(cls.__module__), "OpResolverType", None)
    if enum is None or not hasattr(enum, "BUILTIN_WITHOUT_DEFAULT_DELEGATES"):
        raise RuntimeError("this TFLite backend cannot disable the default (XNNPACK) delegate")
    return enum.BUILTIN_WITHOUT_DEFAULT_DELEGATES


class TFLiteAutoencoder:
    """Runs a (1, W, C) -> (1, W, C) autoencoder .tflite, float32 or full-int8 I/O.

    xnnpack=False asks the runtime not to apply its default delegate (for A/B
    latency comparison); raises RuntimeError if the backend cannot do that.
    """

    def __init__(self, model_path: str | Path | None = None,
                 model_content: bytes | None = None, num_threads: int = 1,
                 xnnpack: bool = True) -> None:
        if (model_path is None) == (model_content is None):
            raise ValueError("give exactly one of model_path or model_content")
        cls, self.backend = _interpreter_class()
        kwargs: dict = {"num_threads": num_threads}
        if model_path is not None:
            kwargs["model_path"] = str(model_path)
        else:
            kwargs["model_content"] = bytes(model_content)
        if not xnnpack:
            kwargs["experimental_op_resolver_type"] = _no_default_delegates(cls)
        try:
            self._it = cls(**kwargs)
        except TypeError as exc:
            raise RuntimeError(f"interpreter rejected options {sorted(kwargs)}: {exc}") from exc
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
        self._out_quantized = bool(np.issubdtype(out["dtype"], np.integer))

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

    # -- one window: prepare -> invoke -> collect -------------------------------
    def _feed(self, x1: np.ndarray) -> np.ndarray:
        return self._quantize(x1) if self.is_quantized else x1

    def _collect(self) -> np.ndarray:
        y = self._it.get_tensor(self._out["index"])
        if self._out_quantized:
            scale, zp = self._out["quantization"]
            return (y.astype(np.float32) - zp) * scale
        return y.copy()

    @staticmethod
    def _error(x: np.ndarray, recon: np.ndarray) -> float:
        return float(((recon - x) ** 2).mean(dtype=np.float64))

    # -- public API -----------------------------------------------------------------
    def reconstruct(self, windows: np.ndarray) -> np.ndarray:
        """Reconstruct windows one at a time (batch 1, as deployed). Returns float32."""
        w = self._check(windows)
        out = np.empty_like(w)
        for i in range(w.shape[0]):
            self._it.set_tensor(self._in["index"], self._feed(w[i : i + 1]))
            self._it.invoke()
            out[i] = self._collect()[0]
        return out

    def reconstruction_errors(self, windows: np.ndarray) -> np.ndarray:
        """Per-window MSE against the ORIGINAL float window, shape (N,)."""
        w = self._check(windows)
        errs = np.empty(w.shape[0], dtype=np.float64)
        for i in range(w.shape[0]):
            self._it.set_tensor(self._in["index"], self._feed(w[i : i + 1]))
            self._it.invoke()
            errs[i] = self._error(w[i], self._collect()[0])
        return errs

    def timed_errors(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like reconstruction_errors, plus per-window timings in nanoseconds.

        Returns (errors, invoke_ns, total_ns). invoke_ns covers interpreter.invoke() only;
        total_ns covers quantize + set_tensor + invoke + get_tensor + dequantize + error.
        Timed from Python, so it includes binding overhead.
        """
        w = self._check(windows)
        n = w.shape[0]
        errs = np.empty(n, dtype=np.float64)
        inv = np.empty(n, dtype=np.int64)
        tot = np.empty(n, dtype=np.int64)
        for i in range(n):
            t0 = perf_counter_ns()
            self._it.set_tensor(self._in["index"], self._feed(w[i : i + 1]))
            t1 = perf_counter_ns()
            self._it.invoke()
            t2 = perf_counter_ns()
            errs[i] = self._error(w[i], self._collect()[0])
            t3 = perf_counter_ns()
            inv[i], tot[i] = t2 - t1, t3 - t0
        return errs, inv, tot

    def describe(self) -> dict:
        def q(d):
            scale, zp = d["quantization"]
            return {"dtype": np.dtype(d["dtype"]).name, "scale": float(scale), "zero_point": int(zp)}
        return {"backend": self.backend, "input_shape": list(self.input_shape),
                "input": q(self._in), "output": q(self._out)}
