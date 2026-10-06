"""Fixed-capacity circular buffer of samples. No allocation after construction
(except the small, necessarily-new array returned by ``window()``)."""
from __future__ import annotations

import numpy as np


class RingBuffer:
    """Holds the most recent ``capacity`` samples of shape (n_channels,)."""

    def __init__(self, capacity: int, n_channels: int, dtype=np.float64) -> None:
        if capacity < 1 or n_channels < 1:
            raise ValueError("capacity and n_channels must be >= 1")
        self.capacity = capacity
        self.n_channels = n_channels
        self._buf = np.empty((capacity, n_channels), dtype=dtype)
        self._write = 0      # next slot to write
        self._count = 0      # valid samples currently stored, capped at capacity
        self.total_pushed = 0

    @property
    def is_full(self) -> bool:
        return self._count == self.capacity

    def push(self, sample) -> None:
        """Overwrite the oldest sample (circular). O(1), writes into the preallocated buffer."""
        s = np.asarray(sample, dtype=self._buf.dtype)
        if s.shape != (self.n_channels,):
            raise ValueError(f"expected shape ({self.n_channels},), got {s.shape}")
        self._buf[self._write] = s
        self._write = (self._write + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)
        self.total_pushed += 1

    def window(self) -> np.ndarray:
        """Oldest-to-newest view of the buffer as a new (capacity, n_channels) array."""
        if not self.is_full:
            raise RuntimeError(f"buffer has only {self._count}/{self.capacity} samples")
        if self._write == 0:
            return self._buf.copy()
        return np.concatenate([self._buf[self._write :], self._buf[: self._write]])

    def reset(self) -> None:
        self._write = 0
        self._count = 0
        self.total_pushed = 0
