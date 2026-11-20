"""
Unit tests for SquatRepCounter (STEP 3.2).

Tests:
1. Valid single repetition
2. Valid multiple repetitions
3. Repeated state updates (idempotency, no duplicate counts)
4. Partial movements (reversing before BOTTOM)
5. Skipped transitions
6. Conservative tracking loss (UNKNOWN) handling
7. Reversal mid-ascent (dipping back down)
8. Reset functionality
9. Starting in non-standing state
"""

import sys
from pathlib import Path

# Ensure `doc` directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from squat_stage import SquatStage, SquatStageResult
from squat_counter import SquatRepCounter, RepState


def test_valid_single_repetition():
    counter = SquatRepCounter()
    sequence = [
        SquatStage.STANDING,
        SquatStage.DESCENDING,
        SquatStage.BOTTOM,
        SquatStage.ASCENDING,
        SquatStage.STANDING,
    ]
    for stage in sequence[:-1]:
        res = counter.update(stage)
        assert res.reps == 0
        assert not res.rep_completed

    res = counter.update(sequence[-1])
    assert res.reps == 1
    assert res.rep_completed
    assert counter.reps == 1


def test_valid_multiple_repetitions():
    counter = SquatRepCounter()
    counter.update(SquatStage.STANDING)

    for expected_rep in range(1, 4):
        counter.update(SquatStage.DESCENDING)
        counter.update(SquatStage.BOTTOM)
        counter.update(SquatStage.ASCENDING)
        res = counter.update(SquatStage.STANDING)
        assert res.reps == expected_rep
        assert res.rep_completed
    assert counter.reps == 3


def test_repeated_states_do_not_duplicate_counts():
    counter = SquatRepCounter()
    # Many frames in STANDING
    for _ in range(10):
        res = counter.update(SquatStage.STANDING)
        assert res.reps == 0

    # Many frames in DESCENDING
    for _ in range(15):
        res = counter.update(SquatStage.DESCENDING)
        assert res.reps == 0
        assert res.is_counting

    # Many frames in BOTTOM
    for _ in range(10):
        res = counter.update(SquatStage.BOTTOM)
        assert res.reps == 0
        assert res.is_counting

    # Many frames in ASCENDING
    for _ in range(12):
        res = counter.update(SquatStage.ASCENDING)
        assert res.reps == 0
        assert res.is_counting

    # First frame in STANDING increments rep
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 1
    assert res.rep_completed
    assert not res.is_counting

    # Subsequent frames in STANDING do NOT increment rep
    for _ in range(20):
        res = counter.update(SquatStage.STANDING)
        assert res.reps == 1
        assert not res.rep_completed


def test_partial_squat_no_bottom_does_not_count():
    counter = SquatRepCounter()
    counter.update(SquatStage.STANDING)

    # User descends partially, then reverses upward without reaching BOTTOM
    counter.update(SquatStage.DESCENDING)
    counter.update(SquatStage.ASCENDING)
    res = counter.update(SquatStage.STANDING)

    assert res.reps == 0
    assert not res.rep_completed
    assert counter.reps == 0

    # Next full squat should count cleanly
    counter.update(SquatStage.DESCENDING)
    counter.update(SquatStage.BOTTOM)
    counter.update(SquatStage.ASCENDING)
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 1


def test_skipped_transitions_do_not_count():
    counter = SquatRepCounter()
    counter.update(SquatStage.STANDING)

    # Jump straight to BOTTOM (skipping DESCENDING)
    counter.update(SquatStage.BOTTOM)
    counter.update(SquatStage.ASCENDING)
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 0

    # Jump straight from BOTTOM to STANDING (skipping ASCENDING)
    counter.update(SquatStage.STANDING)
    counter.update(SquatStage.DESCENDING)
    counter.update(SquatStage.BOTTOM)
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 0


def test_tracking_loss_unknown_aborts_rep():
    counter = SquatRepCounter()
    counter.update(SquatStage.STANDING)
    counter.update(SquatStage.DESCENDING)
    counter.update(SquatStage.BOTTOM)

    # Tracking lost at bottom
    res = counter.update(SquatStage.UNKNOWN)
    assert res.reps == 0
    assert not res.is_counting

    # Camera recovers during ascent
    counter.update(SquatStage.ASCENDING)
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 0  # Aborted! Uncertain sequence.


def test_reversal_mid_ascent_does_not_count():
    counter = SquatRepCounter()
    counter.update(SquatStage.STANDING)
    counter.update(SquatStage.DESCENDING)
    counter.update(SquatStage.BOTTOM)
    counter.update(SquatStage.ASCENDING)

    # Dipped back down mid-ascent
    counter.update(SquatStage.DESCENDING)
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 0


def test_reset_functionality():
    counter = SquatRepCounter()
    counter.update(SquatStage.STANDING)
    counter.update(SquatStage.DESCENDING)
    counter.update(SquatStage.BOTTOM)
    counter.update(SquatStage.ASCENDING)
    counter.update(SquatStage.STANDING)
    assert counter.reps == 1

    # Reset
    counter.reset()
    assert counter.reps == 0
    assert not counter.is_counting

    # Recovers cleanly when STANDING is confirmed
    counter.update(SquatStage.STANDING)
    counter.update(SquatStage.DESCENDING)
    counter.update(SquatStage.BOTTOM)
    counter.update(SquatStage.ASCENDING)
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 1


def test_starts_in_non_standing_does_not_count_until_first_standing():
    counter = SquatRepCounter()
    # Program starts while user is already at BOTTOM
    counter.update(SquatStage.BOTTOM)
    counter.update(SquatStage.ASCENDING)
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 0  # Did not start from STANDING!

    # Now that STANDING was achieved, next full rep counts
    counter.update(SquatStage.DESCENDING)
    counter.update(SquatStage.BOTTOM)
    counter.update(SquatStage.ASCENDING)
    res = counter.update(SquatStage.STANDING)
    assert res.reps == 1


def test_accepts_squat_stage_result_object():
    counter = SquatRepCounter()
    sr_standing = SquatStageResult(
        stage=SquatStage.STANDING,
        knee_angle=170.0,
        hip_angle=168.0,
        knee_velocity=0.0,
        hip_velocity=0.0,
        source="both",
    )
    sr_descending = SquatStageResult(
        stage=SquatStage.DESCENDING,
        knee_angle=140.0,
        hip_angle=135.0,
        knee_velocity=-25.0,
        hip_velocity=-25.0,
        source="both",
    )
    sr_bottom = SquatStageResult(
        stage=SquatStage.BOTTOM,
        knee_angle=105.0,
        hip_angle=110.0,
        knee_velocity=0.0,
        hip_velocity=0.0,
        source="both",
    )
    sr_ascending = SquatStageResult(
        stage=SquatStage.ASCENDING,
        knee_angle=145.0,
        hip_angle=148.0,
        knee_velocity=25.0,
        hip_velocity=25.0,
        source="both",
    )

    counter.update(sr_standing)
    counter.update(sr_descending)
    counter.update(sr_bottom)
    counter.update(sr_ascending)
    res = counter.update(sr_standing)
    assert res.reps == 1


if __name__ == "__main__":
    test_valid_single_repetition()
    test_valid_multiple_repetitions()
    test_repeated_states_do_not_duplicate_counts()
    test_partial_squat_no_bottom_does_not_count()
    test_skipped_transitions_do_not_count()
    test_tracking_loss_unknown_aborts_rep()
    test_reversal_mid_ascent_does_not_count()
    test_reset_functionality()
    test_starts_in_non_standing_does_not_count_until_first_standing()
    test_accepts_squat_stage_result_object()
    print("ALL SQUAT REP COUNTER TESTS PASSED!")
