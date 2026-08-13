"""Stable extension interfaces for future SeRT or experimental integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Mapping, Sequence


class CommandProvider(ABC):
    @abstractmethod
    def command(self, timestamp_s: float) -> Sequence[float]:
        """Return ``[vx, vy, yaw_rate]`` for the current control step."""


@dataclass
class ConstantCommandProvider(CommandProvider):
    value: Sequence[float] = (0.0, 0.0, 0.0)

    def command(self, timestamp_s: float) -> Sequence[float]:
        del timestamp_s
        if len(self.value) != 3:
            raise ValueError("velocity command must have three values")
        return tuple(float(item) for item in self.value)


class KeyboardCommandProvider(CommandProvider):
    """Adapter around a UI-owned key-state callback; it does not capture stdin."""

    def __init__(self, key_state: Callable[[], Mapping[str, bool]], speed: float = 0.3):
        self._key_state = key_state
        self.speed = float(speed)

    def command(self, timestamp_s: float) -> Sequence[float]:
        del timestamp_s
        keys = self._key_state()
        return (
            self.speed * (float(keys.get("up", False)) - float(keys.get("down", False))),
            self.speed * (float(keys.get("left", False)) - float(keys.get("right", False))),
            self.speed * (float(keys.get("q", False)) - float(keys.get("e", False))),
        )


class GamepadCommandProvider(CommandProvider):
    """Adapter around a callback returning normalized ``(vx, vy, yaw)`` axes."""

    def __init__(self, axes: Callable[[], Sequence[float]], scale: Sequence[float] = (1.0, 0.3, 0.2)):
        self._axes = axes
        self._scale = tuple(float(item) for item in scale)

    def command(self, timestamp_s: float) -> Sequence[float]:
        del timestamp_s
        values = tuple(float(item) for item in self._axes())
        if len(values) != 3:
            raise ValueError("gamepad callback must return three axes")
        return tuple(value * scale for value, scale in zip(values, self._scale))


class ScriptedCommandProvider(CommandProvider):
    def __init__(self, timeline: Sequence[tuple[float, Sequence[float]]]):
        if not timeline:
            raise ValueError("scripted command timeline cannot be empty")
        self.timeline = sorted((float(t), tuple(float(v) for v in command)) for t, command in timeline)
        if any(len(command) != 3 for _, command in self.timeline):
            raise ValueError("every scripted command must contain three values")

    def command(self, timestamp_s: float) -> Sequence[float]:
        selected = self.timeline[0][1]
        for start, value in self.timeline:
            if timestamp_s < start:
                break
            selected = value
        return selected


class ExternalPolicyCommandProvider(CommandProvider):
    """Thread-safe bridge for a future high-level SeRT/SERT process."""

    def __init__(self):
        self._lock = Lock()
        self._value = (0.0, 0.0, 0.0)
        self._timestamp_s: float | None = None

    def update(self, command: Sequence[float], timestamp_s: float) -> None:
        value = tuple(float(item) for item in command)
        if len(value) != 3:
            raise ValueError("external command must contain three values")
        with self._lock:
            self._value = value
            self._timestamp_s = float(timestamp_s)

    def command(self, timestamp_s: float) -> Sequence[float]:
        del timestamp_s
        with self._lock:
            return self._value


class SafetyFilter(ABC):
    @abstractmethod
    def filter(
        self,
        *,
        observation: Mapping[str, object],
        nominal_command: Sequence[float],
        timestamp: float,
    ) -> tuple[Sequence[float], Mapping[str, object]]:
        """Return safe high-level command and diagnostic information."""


class IdentitySafetyFilter(SafetyFilter):
    def filter(
        self,
        *,
        observation: Mapping[str, object],
        nominal_command: Sequence[float],
        timestamp: float,
    ) -> tuple[Sequence[float], Mapping[str, object]]:
        del observation, timestamp
        return list(nominal_command), {"modified": False}


class SensorExtension(ABC):
    @abstractmethod
    def observations(self, timestamp_s: float) -> Mapping[str, Sequence[float]]:
        """Return named optional observation terms."""


class SensorRegistry:
    def __init__(self):
        self._extensions: dict[str, SensorExtension] = {}

    def register(self, name: str, extension: SensorExtension) -> None:
        if not name or name in self._extensions:
            raise ValueError(f"invalid or duplicate sensor extension: {name!r}")
        self._extensions[name] = extension

    def collect(self, timestamp_s: float) -> dict[str, Sequence[float]]:
        result: dict[str, Sequence[float]] = {}
        for prefix, extension in self._extensions.items():
            for name, value in extension.observations(timestamp_s).items():
                key = f"{prefix}.{name}"
                if key in result:
                    raise ValueError(f"duplicate sensor observation: {key}")
                result[key] = value
        return result
