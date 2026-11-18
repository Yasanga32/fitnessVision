"""
Lightweight temporal smoothing for real-time joint angles.

Uses an Exponential Moving Average (EMA) per joint to damp frame-to-frame
landmark jitter while keeping latency low, and supports resetting individual
joints or all joints whenever tracking is lost so stale values are never reused.
"""

from typing import Dict, Optional


class AngleSmoother:
    """Per-joint Exponential Moving Average (EMA) smoother with tracking-loss reset."""

    def __init__(self, alpha: float = 0.45) -> None:
        """
        Args:
            alpha: Smoothing factor in (0.0, 1.0]. Higher values follow rapid
                motion with less delay; lower values smooth jitter more strongly.
                Default 0.45 balances low lag (~1.2 frames) with jitter reduction.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0.0, 1.0].")
        self.alpha = alpha
        self._values: Dict[str, float] = {}

    def update(self, key: str, raw_angle: float) -> float:
        """Update and return the smoothed angle for `key`."""
        prev = self._values.get(key)
        if prev is None:
            smoothed = raw_angle
        else:
            smoothed = self.alpha * raw_angle + (1.0 - self.alpha) * prev
        self._values[key] = smoothed
        return smoothed

    def get(self, key: str) -> Optional[float]:
        """Return the current smoothed value for `key`, or None if unset/reset."""
        return self._values.get(key)

    def reset(self, key: str) -> None:
        """Clear the smoothed state for one joint after tracking loss."""
        self._values.pop(key, None)

    def reset_all(self) -> None:
        """Clear all smoothed joint states (e.g., when no pose is detected)."""
        self._values.clear()
