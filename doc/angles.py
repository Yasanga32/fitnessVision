"""
Reusable 2D joint angle calculation utilities for MediaPipe Pose landmarks.

Designed to be extended easily to other joints (e.g., right elbow, knees, hips)
by passing any `(point_a, point_b, point_c)` landmark index triplet.
"""

import math
from typing import Sequence, Tuple, Optional

Point2D = Tuple[float, float]

# MediaPipe Pose landmark indices for the left arm
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

# Joint triplet ordered as (proximal, vertex/joint, distal)
LEFT_ELBOW_JOINT = (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)


def calculate_angle(point_a: Point2D, point_b: Point2D, point_c: Point2D) -> float:
    """
    Calculate the 2D interior angle (in degrees, 0-180) at `point_b`
    formed by the segments BA and BC.

    Args:
        point_a: (x, y) pixel coordinates of the first point (e.g., shoulder).
        point_b: (x, y) pixel coordinates of the joint vertex (e.g., elbow).
        point_c: (x, y) pixel coordinates of the third point (e.g., wrist).

    Returns:
        Angle in degrees in the range [0.0, 180.0]. Returns 0.0 if either
        segment has zero length.
    """
    # Build 2D vectors pointing from the joint (B) toward A and C.
    ba_x, ba_y = point_a[0] - point_b[0], point_a[1] - point_b[1]
    bc_x, bc_y = point_c[0] - point_b[0], point_c[1] - point_b[1]

    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)
    if mag_ba == 0.0 or mag_bc == 0.0:
        return 0.0

    # Dot product formula: cos(theta) = (BA . BC) / (|BA| * |BC|)
    cos_angle = (ba_x * bc_x + ba_y * bc_y) / (mag_ba * mag_bc)
    # Clamp to [-1.0, 1.0] to guard against floating-point rounding outside domain.
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def get_joint_angle(
    pose: Sequence,
    joint_indices: Tuple[int, int, int],
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.5,
) -> Optional[Tuple[float, Tuple[int, int]]]:
    """
    Compute a 2D joint angle in degrees from pixel-scaled MediaPipe landmarks,
    only when all three landmarks meet the visibility threshold.

    Args:
        pose: Sequence of 33 MediaPipe pose landmarks.
        joint_indices: (idx_a, idx_b, idx_c) where idx_b is the joint vertex.
        frame_width: Width of the video frame in pixels.
        frame_height: Height of the video frame in pixels.
        min_visibility: Minimum required landmark visibility score (0.0-1.0).

    Returns:
        (angle_degrees, (vertex_x, vertex_y)) if all three landmarks are
        sufficiently visible, otherwise None.
    """
    idx_a, idx_b, idx_c = joint_indices
    lm_a, lm_b, lm_c = pose[idx_a], pose[idx_b], pose[idx_c]

    for lm in (lm_a, lm_b, lm_c):
        visibility = getattr(lm, "visibility", None)
        if visibility is None or visibility < min_visibility:
            return None

    # Scale normalized coordinates [0.0, 1.0] into pixel space so aspect ratio
    # does not distort the 2D angle.
    point_a = (lm_a.x * frame_width, lm_a.y * frame_height)
    point_b = (lm_b.x * frame_width, lm_b.y * frame_height)
    point_c = (lm_c.x * frame_width, lm_c.y * frame_height)

    angle = calculate_angle(point_a, point_b, point_c)
    vertex_px = (int(point_b[0]), int(point_b[1]))
    return angle, vertex_px
