"""
Squat repetition counter based on sequential movement stages.

Counts exactly one repetition for the complete, uninterrupted sequence:
    STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING

Independent of OpenCV and camera capture for modularity and testability.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from squat_stage import SquatStage, SquatStageResult


class RepState(str, Enum):
    """Internal states tracking the progression through a single repetition."""

    IDLE = "IDLE"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"


@dataclass
class SquatCounterResult:
    """Snapshot of repetition counter state for the current frame."""

    reps: int
    stage: SquatStage
    is_counting: bool
    rep_completed: bool = False  # True only on the exact frame a rep completes


class SquatRepCounter:
    """
    State machine that counts completed squat repetitions.

    Enforces the strict movement cycle:
        STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING

    Rules:
    - Must start from confirmed STANDING (is_ready=True).
    - Ignores repeated frames in the same state (idempotent).
    - Aborts rep progression on partial movements (e.g. DESCENDING -> ASCENDING without BOTTOM).
    - Aborts rep progression on skipped transitions (e.g. STANDING -> BOTTOM).
    - Resets rep progression immediately on UNKNOWN / tracking loss.
    - Increments rep count exactly once upon transition from ASCENDING -> STANDING.
    """

    def __init__(self) -> None:
        self._reps: int = 0
        self._state: RepState = RepState.IDLE
        self._is_ready: bool = False
        self._last_stage: Optional[SquatStage] = None

    @property
    def reps(self) -> int:
        """Total number of completed repetitions."""
        return self._reps

    @property
    def is_counting(self) -> bool:
        """True if currently in the middle of an active repetition sequence."""
        return self._state != RepState.IDLE

    @property
    def rep_state(self) -> RepState:
        """Current internal sequence state."""
        return self._state

    def reset(self) -> None:
        """Reset the repetition count to zero and clear in-progress state."""
        self._reps = 0
        self._state = RepState.IDLE
        self._is_ready = False
        self._last_stage = None

    def update(
        self, stage_input: Union[SquatStage, SquatStageResult]
    ) -> SquatCounterResult:
        """
        Process the current squat stage and update rep count if a valid
        cycle completes.
        """
        if isinstance(stage_input, SquatStageResult):
            stage = stage_input.stage
        else:
            stage = stage_input

        rep_completed = False

        # Conservative tracking-loss handling: any UNKNOWN immediately aborts progress
        if stage == SquatStage.UNKNOWN:
            self._state = RepState.IDLE
            self._is_ready = False
            self._last_stage = SquatStage.UNKNOWN
            return SquatCounterResult(
                reps=self._reps,
                stage=stage,
                is_counting=False,
                rep_completed=False,
            )

        # Detect state transitions (ignore repeated state updates)
        is_transition = stage != self._last_stage
        self._last_stage = stage

        if is_transition:
            if stage == SquatStage.STANDING:
                if self._state == RepState.ASCENDING:
                    # Completed full sequence: STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING
                    self._reps += 1
                    rep_completed = True
                # User is in STANDING, ready for a new rep
                self._state = RepState.IDLE
                self._is_ready = True

            elif stage == SquatStage.DESCENDING:
                if self._is_ready and self._state == RepState.IDLE:
                    self._state = RepState.DESCENDING
                else:
                    # Invalid sequence (e.g. descending from ASCENDING or without prior STANDING)
                    self._state = RepState.IDLE
                    self._is_ready = False

            elif stage == SquatStage.BOTTOM:
                if self._state == RepState.DESCENDING:
                    self._state = RepState.BOTTOM
                else:
                    # Skipped DESCENDING or unexpected jump
                    self._state = RepState.IDLE
                    self._is_ready = False

            elif stage == SquatStage.ASCENDING:
                if self._state == RepState.BOTTOM:
                    self._state = RepState.ASCENDING
                else:
                    # Incomplete movement (e.g. reversed upward from DESCENDING before hitting BOTTOM)
                    self._state = RepState.IDLE
                    self._is_ready = False

        return SquatCounterResult(
            reps=self._reps,
            stage=stage,
            is_counting=(self._state != RepState.IDLE),
            rep_completed=rep_completed,
        )
