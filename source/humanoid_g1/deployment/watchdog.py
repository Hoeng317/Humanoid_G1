"""Monotonic packet and loop watchdog."""

from __future__ import annotations

import time


class Watchdog:
    def __init__(self, timeout_s: float):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = float(timeout_s)
        self._last_kick: float | None = None

    def kick(self, timestamp_s: float | None = None) -> None:
        self._last_kick = time.monotonic() if timestamp_s is None else float(timestamp_s)

    def expired(self, timestamp_s: float | None = None) -> bool:
        if self._last_kick is None:
            return True
        now = time.monotonic() if timestamp_s is None else float(timestamp_s)
        return now - self._last_kick > self.timeout_s

