"""
Rule-based Squat Movement Stage Detection state machine.

This module is intentionally independent of OpenCV and camera capture: it
consumes already-smoothed, reliability-filtered knee and hip angles together
with a monotonic timestamp and outputs the current movement stage.

NOTE: All angle thresholds in `SquatStageConfig` are initial experimental
starting values for movement-phase segmentation only. They require calibration
per camera angle and user proportions and must NOT be interpreted as proof of
correct or safe squat technique.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional, Tuple


class SquatStage(str, Enum):
    """Supported squat movement stages."""

    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"
    UNKNOWN = "UNKNOWN"


@dataclass
class SquatStageConfig:
    """
    Configurable initial experimental thresholds for squat stage detection.

    Hysteresis gaps between `*_enter` and `*_exit` thresholds, combined with
    `min_state_duration_sec`, prevent rapid flickering near boundaries.
    """

    # Standing entry/exit thresholds (degrees) — 10 deg hysteresis band
    standing_knee_enter: float = 160.0
    standing_hip_enter: float = 155.0
    standing_knee_exit: float = 150.0
    standing_hip_exit: float = 145.0

    # Bottom entry/exit thresholds (degrees) — 10 deg hysteresis band
    bottom_knee_enter: float = 115.0
    bottom_hip_enter: float = 125.0
    bottom_knee_exit: float = 125.0
    bottom_hip_exit: float = 135.0

    # Extra angular margin allowing very slow controlled transitions even when
    # instantaneous angular velocity stays below `min_velocity_deg_per_sec`.
    slow_transition_margin_deg: float = 8.0

    # Minimum combined angle change rate (deg/sec) to classify active motion
    min_velocity_deg_per_sec: float = 12.0

    # Lookback window (seconds) used to estimate smooth angular velocity
    velocity_window_sec: float = 0.10

    # Minimum continuous time (seconds) a candidate transition must hold
    min_state_duration_sec: float = 0.10


@dataclass
class SquatStageResult:
    """Output snapshot produced on each update of `SquatStageDetector`."""

    stage: SquatStage
    knee_angle: Optional[float]
    hip_angle: Optional[float]
    knee_velocity: float
    hip_velocity: float
    source: str  # "both", "left", "right", or "none"


class SquatStageDetector:
    """
    Rule-based state machine that classifies squat movement stages from
    reliable, smoothed knee and hip angles over time.
    """

    def __init__(self, config: Optional[SquatStageConfig] = None) -> None:
        self.config = config or SquatStageConfig()
        self._stage: SquatStage = SquatStage.UNKNOWN
        self._pending_stage: Optional[SquatStage] = None
        self._pending_since: Optional[float] = None
        self._active_source: str = "none"
        # Rolling samples of (timestamp_sec, knee_angle, hip_angle)
        self._history: Deque[Tuple[float, float, float]] = deque()

    @property
    def stage(self) -> SquatStage:
        """Current confirmed squat movement stage."""
        return self._stage

    def reset(self) -> None:
        """Reset state machine to UNKNOWN and clear angle history."""
        self._stage = SquatStage.UNKNOWN
        self._pending_stage = None
        self._pending_since = None
        self._active_source = "none"
        self._history.clear()

    @staticmethod
    def _select_reliable_pair(
        left_knee: Optional[float],
        right_knee: Optional[float],
        left_hip: Optional[float],
        right_hip: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        Combine reliable knee and hip angles without mixing mismatched sides.

        - If both sides have complete reliable (knee, hip) pairs, average them.
        - If only one side has a complete reliable (knee, hip) pair, use it.
        - Otherwise return (None, None, "none") to indicate insufficient tracking.
        """
        left_ready = left_knee is not None and left_hip is not None
        right_ready = right_knee is not None and right_hip is not None

        if left_ready and right_ready:
            return (
                0.5 * (left_knee + right_knee),
                0.5 * (left_hip + right_hip),
                "both",
            )
        if left_ready:
            return left_knee, left_hip, "left"
        if right_ready:
            return right_knee, right_hip, "right"
        return None, None, "none"

    def _compute_velocities(
        self,
        timestamp_sec: float,
        knee: float,
        hip: float,
    ) -> Tuple[float, float]:
        """Estimate knee and hip angular velocity (deg/sec) over the lookback window."""
        self._history.append((timestamp_sec, knee, hip))

        # Keep samples within 2x the lookback window (and at least the oldest anchor)
        cutoff = timestamp_sec - max(self.config.velocity_window_sec * 2.0, 0.05)
        while len(self._history) > 2 and self._history[0][0] < cutoff:
            self._history.popleft()

        # Pick the sample closest to `timestamp_sec - velocity_window_sec`
        target_time = timestamp_sec - self.config.velocity_window_sec
        ref_t, ref_knee, ref_hip = self._history[0]
        for sample in self._history:
            if sample[0] <= target_time:
                ref_t, ref_knee, ref_hip = sample
            else:
                break

        dt = timestamp_sec - ref_t
        if dt <= 1e-6:
            return 0.0, 0.0
        return (knee - ref_knee) / dt, (hip - ref_hip) / dt

    def _evaluate_candidate(
        self,
        knee: float,
        hip: float,
        knee_vel: float,
        hip_vel: float,
    ) -> SquatStage:
        """Determine the candidate next stage from angles, hysteresis, and motion."""
        cfg = self.config
        combined_vel = 0.5 * (knee_vel + hip_vel)
        moving_down = combined_vel <= -cfg.min_velocity_deg_per_sec
        moving_up = combined_vel >= cfg.min_velocity_deg_per_sec

        is_standing_pose = (
            knee >= cfg.standing_knee_enter and hip >= cfg.standing_hip_enter
        )
        is_bottom_pose = (
            knee <= cfg.bottom_knee_enter and hip <= cfg.bottom_hip_enter
        )

        if self._stage == SquatStage.UNKNOWN:
            if is_standing_pose:
                return SquatStage.STANDING
            if is_bottom_pose:
                return SquatStage.BOTTOM
            if moving_down:
                return SquatStage.DESCENDING
            if moving_up:
                return SquatStage.ASCENDING
            return SquatStage.UNKNOWN

        if self._stage == SquatStage.STANDING:
            exited_standing = (
                knee <= cfg.standing_knee_exit or hip <= cfg.standing_hip_exit
            )
            deeply_exited = (
                knee <= cfg.standing_knee_exit - cfg.slow_transition_margin_deg
                and hip <= cfg.standing_hip_exit - cfg.slow_transition_margin_deg
            )
            if exited_standing and (moving_down or deeply_exited):
                return SquatStage.DESCENDING
            return SquatStage.STANDING

        if self._stage == SquatStage.DESCENDING:
            if is_bottom_pose:
                return SquatStage.BOTTOM
            if is_standing_pose:
                return SquatStage.STANDING
            # Incomplete squat: user reversed upward before reaching BOTTOM
            if moving_up:
                return SquatStage.ASCENDING
            return SquatStage.DESCENDING

        if self._stage == SquatStage.BOTTOM:
            exited_bottom = (
                knee >= cfg.bottom_knee_exit or hip >= cfg.bottom_hip_exit
            )
            clearly_exited = (
                knee >= cfg.bottom_knee_exit + cfg.slow_transition_margin_deg
                and hip >= cfg.bottom_hip_exit + cfg.slow_transition_margin_deg
            )
            if exited_bottom and (moving_up or clearly_exited):
                return SquatStage.ASCENDING
            return SquatStage.BOTTOM

        if self._stage == SquatStage.ASCENDING:
            if is_standing_pose:
                return SquatStage.STANDING
            if is_bottom_pose:
                return SquatStage.BOTTOM
            # Re-descending mid-ascent before returning to STANDING
            if moving_down:
                return SquatStage.DESCENDING
            return SquatStage.ASCENDING

        return SquatStage.UNKNOWN

    def update(
        self,
        left_knee: Optional[float],
        right_knee: Optional[float],
        left_hip: Optional[float],
        right_hip: Optional[float],
        timestamp_sec: float,
    ) -> SquatStageResult:
        """
        Update the squat stage state machine for the current frame.

        Pass `None` for any joint angle whose landmarks are missing or unreliable.
        When tracking is insufficient, immediately outputs `SquatStage.UNKNOWN`
        and makes no new movement decisions.
        """
        knee, hip, source = self._select_reliable_pair(
            left_knee, right_knee, left_hip, right_hip
        )

        # Insufficient tracking -> immediately report UNKNOWN and clear history
        if knee is None or hip is None:
            self.reset()
            return SquatStageResult(
                stage=SquatStage.UNKNOWN,
                knee_angle=None,
                hip_angle=None,
                knee_velocity=0.0,
                hip_velocity=0.0,
                source="none",
            )

        # If the reliable measurement source changed (e.g. "both" -> "left"),
        # clear velocity history so a small L/R offset doesn't spike velocity.
        if source != self._active_source:
            self._history.clear()
            self._active_source = source

        knee_vel, hip_vel = self._compute_velocities(timestamp_sec, knee, hip)
        candidate = self._evaluate_candidate(knee, hip, knee_vel, hip_vel)

        if candidate == self._stage:
            self._pending_stage = None
            self._pending_since = None
        else:
            if candidate != self._pending_stage:
                self._pending_stage = candidate
                self._pending_since = timestamp_sec
            elif (
                self._pending_since is not None
                and (timestamp_sec - self._pending_since)
                >= self.config.min_state_duration_sec - 1e-9
            ):
                self._stage = candidate
                self._pending_stage = None
                self._pending_since = None

        return SquatStageResult(
            stage=self._stage,
            knee_angle=knee,
            hip_angle=hip,
            knee_velocity=knee_vel,
            hip_velocity=hip_vel,
            source=source,
        )
