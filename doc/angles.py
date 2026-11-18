"""
Reusable 2D joint angle calculation and reliability diagnostics for MediaPipe
Pose landmarks.

Designed to be extended easily to other joints (elbows, hips, knees, etc.)
by passing any `(point_a, point_b, point_c)` landmark index triplet.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

Point2D = Tuple[float, float]

# MediaPipe Pose landmark indices for both arms
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

# MediaPipe Pose landmark indices for both legs
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28

# Joint triplets ordered as (proximal, vertex/joint, distal)
LEFT_ELBOW_JOINT = (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
RIGHT_ELBOW_JOINT = (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
LEFT_HIP_JOINT = (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
RIGHT_HIP_JOINT = (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
LEFT_KNEE_JOINT = (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
RIGHT_KNEE_JOINT = (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)


@dataclass
class JointReliability:
    """Diagnostic summary describing whether a 2D joint angle is reliable."""

    is_reliable: bool
    reasons: List[str]
    proximal_dist_px: float
    vertex_dist_px: float
    upper_seg_px: float
    lower_seg_px: float


# Backward-compatible alias for earlier code referencing ElbowReliability
ElbowReliability = JointReliability


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


def is_landmark_in_frame(landmark, margin: float = 0.0) -> bool:
    """Return True if a landmark's normalized (x, y) lies inside the camera frame."""
    return (-margin <= landmark.x <= 1.0 + margin) and (-margin <= landmark.y <= 1.0 + margin)


def get_joint_angle(
    pose: Sequence,
    joint_indices: Tuple[int, int, int],
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.5,
) -> Optional[Tuple[float, Tuple[int, int]]]:
    """
    Compute a 2D joint angle in degrees from pixel-scaled MediaPipe landmarks,
    only when all three landmarks are present, inside the frame, and meet
    the visibility threshold.

    Args:
        pose: Sequence of 33 MediaPipe pose landmarks.
        joint_indices: (idx_a, idx_b, idx_c) where idx_b is the joint vertex.
        frame_width: Width of the video frame in pixels.
        frame_height: Height of the video frame in pixels.
        min_visibility: Minimum required landmark visibility score (0.0-1.0).

    Returns:
        (angle_degrees, (vertex_x, vertex_y)) if all three landmarks are
        sufficiently visible and in frame, otherwise None.
    """
    idx_a, idx_b, idx_c = joint_indices
    lm_a, lm_b, lm_c = pose[idx_a], pose[idx_b], pose[idx_c]

    for lm in (lm_a, lm_b, lm_c):
        visibility = getattr(lm, "visibility", None)
        if visibility is None or visibility < min_visibility:
            return None
        if not is_landmark_in_frame(lm):
            return None

    # Scale normalized coordinates [0.0, 1.0] into pixel space so aspect ratio
    # does not distort the 2D angle.
    point_a = (lm_a.x * frame_width, lm_a.y * frame_height)
    point_b = (lm_b.x * frame_width, lm_b.y * frame_height)
    point_c = (lm_c.x * frame_width, lm_c.y * frame_height)

    angle = calculate_angle(point_a, point_b, point_c)
    vertex_px = (int(point_b[0]), int(point_b[1]))
    return angle, vertex_px


def get_joint_visibilities(
    pose: Sequence,
    joint_indices: Tuple[int, int, int],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return the visibility scores for a (proximal, vertex, distal) triplet."""
    idx_a, idx_b, idx_c = joint_indices
    return (
        getattr(pose[idx_a], "visibility", None),
        getattr(pose[idx_b], "visibility", None),
        getattr(pose[idx_c], "visibility", None),
    )


def evaluate_joint_reliability(
    pose: Sequence,
    selected_joint: Tuple[int, int, int],
    opposite_joint: Tuple[int, int, int],
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.5,
    overlap_width_ratio: float = 0.05,
    side_profile_width_ratio: float = 0.06,
    min_bone_width_ratio: float = 0.03,
) -> JointReliability:
    """
    Evaluate whether a 2D joint angle measurement is reliable.

    Visibility alone does not guarantee accuracy in a monocular camera, so this
    combines multiple independent signals:
    1. All three required landmarks meet `min_visibility`.
    2. All three landmarks lie inside the visible frame (no off-screen guesses).
    3. The selected joint vertex does not directly overlap its opposite-side
       counterpart in 2D pixel space.
    4. In a side-facing profile (proximal pair collapsed in 2D), the selected
       side is not the occluded far side (lower visibility or deeper `z`).
    5. Neither adjacent limb segment is severely foreshortened in 2D.
    """
    sel_a, sel_b, sel_c = (pose[i] for i in selected_joint)
    opp_a, opp_b, opp_c = (pose[i] for i in opposite_joint)

    def to_px(lm) -> Point2D:
        return (lm.x * frame_width, lm.y * frame_height)

    def dist_px(lm1, lm2) -> float:
        p1, p2 = to_px(lm1), to_px(lm2)
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    proximal_dist_px = dist_px(sel_a, opp_a)
    vertex_dist_px = dist_px(sel_b, opp_b)
    upper_seg_px = dist_px(sel_a, sel_b)
    lower_seg_px = dist_px(sel_b, sel_c)

    overlap_threshold_px = frame_width * overlap_width_ratio
    side_profile_threshold_px = frame_width * side_profile_width_ratio
    min_bone_px = frame_width * min_bone_width_ratio

    reasons: List[str] = []

    # 1. Missing or low-confidence landmarks on the selected joint
    sel_vis = [getattr(lm, "visibility", None) for lm in (sel_a, sel_b, sel_c)]
    if any(v is None or v < min_visibility for v in sel_vis):
        reasons.append("low visibility")

    # 2. Extrapolated landmarks outside the camera frame
    if any(not is_landmark_in_frame(lm) for lm in (sel_a, sel_b, sel_c)):
        reasons.append("out of frame")

    # 3. Direct 2D overlap between left and right joint vertices
    if vertex_dist_px < overlap_threshold_px:
        reasons.append("L/R overlap")

    # 4. Side-facing profile: flag the occluded far side behind the torso
    if proximal_dist_px < side_profile_threshold_px:
        opp_vis = [getattr(lm, "visibility", None) for lm in (opp_a, opp_b, opp_c)]
        sel_avg_vis = sum(v for v in sel_vis if v is not None) / 3.0
        opp_avg_vis = sum(v for v in opp_vis if v is not None) / 3.0

        sel_z = getattr(sel_b, "z", None)
        opp_z = getattr(opp_b, "z", None)
        is_farther = (
            sel_z is not None and opp_z is not None and (sel_z - opp_z) > 0.05
        )
        if opp_avg_vis - sel_avg_vis > 0.1 or is_farther:
            reasons.append("far-side occluded")

    # 5. Severe 2D foreshortening (limb pointing toward/away from camera)
    if upper_seg_px < min_bone_px or lower_seg_px < min_bone_px:
        reasons.append("foreshortened")

    return JointReliability(
        is_reliable=len(reasons) == 0,
        reasons=reasons,
        proximal_dist_px=proximal_dist_px,
        vertex_dist_px=vertex_dist_px,
        upper_seg_px=upper_seg_px,
        lower_seg_px=lower_seg_px,
    )


# Backward-compatible alias
evaluate_elbow_reliability = evaluate_joint_reliability
