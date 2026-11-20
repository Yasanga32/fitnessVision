"""
Real-time Pose Landmarker using OpenCV + MediaPipe Tasks API.

Pipeline overview (runs once per webcam frame):
1. Capture a BGR frame from the webcam (OpenCV).
2. Mirror it horizontally and convert BGR -> RGB (MediaPipe expects RGB).
3. Wrap the RGB frame in an `mp.Image` and pass it with an increasing
   millisecond timestamp to `PoseLandmarker.detect_for_video(...)`.
4. Convert the returned normalized landmark coordinates (0.0-1.0) into
   pixel coordinates and draw joints (circles) + skeleton bones (lines).
5. Measure and overlay the live FPS, then display the frame until Esc is pressed.
"""

from pathlib import Path          # OS-independent file path handling
import time                       # High-precision timer for FPS calculation
import cv2                        # OpenCV: webcam capture, image ops, drawing, display
import mediapipe as mp            # MediaPipe core (used for mp.Image)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from angles import (
    LEFT_ELBOW_JOINT,
    LEFT_HIP_JOINT,
    LEFT_KNEE_JOINT,
    RIGHT_ELBOW_JOINT,
    RIGHT_HIP_JOINT,
    RIGHT_KNEE_JOINT,
    evaluate_joint_reliability,
    get_joint_angle,
    get_joint_visibilities,
)
from smoothing import AngleSmoother
from squat_counter import SquatRepCounter
from squat_stage import SquatStage, SquatStageDetector

# Temporary debug flag for side-view overlap investigation (press 'D' to toggle live)
DEBUG_SIDE_VIEW = True

# MediaPipe Pose left/right landmark index sets (index 0 = nose/center)
LEFT_LANDMARKS = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT_LANDMARKS = {4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

# Distinct BGR colors used in side-view debug mode
COLOR_LEFT = (255, 140, 0)      # Blue (MediaPipe Left side)
COLOR_RIGHT = (0, 110, 255)     # Orange (MediaPipe Right side)
COLOR_CENTER = (220, 220, 220)  # Light gray (center / cross-body connections)


def get_landmark_color(index: int, debug_mode: bool):
    """Return BGR color for a landmark (side-coded in debug mode, green normally)."""
    if not debug_mode:
        return (0, 255, 0)
    if index in LEFT_LANDMARKS:
        return COLOR_LEFT
    if index in RIGHT_LANDMARKS:
        return COLOR_RIGHT
    return COLOR_CENTER


def get_connection_color(start_idx: int, end_idx: int, debug_mode: bool):
    """Return BGR color for a skeleton bone (side-coded in debug mode, cyan normally)."""
    if not debug_mode:
        return (255, 255, 0)
    if start_idx in LEFT_LANDMARKS and end_idx in LEFT_LANDMARKS:
        return COLOR_LEFT
    if start_idx in RIGHT_LANDMARKS and end_idx in RIGHT_LANDMARKS:
        return COLOR_RIGHT
    return COLOR_CENTER


# All tracked joints: (label_prefix, joint_triplet, opposite_triplet, align_right_of_joint)
TRACKED_JOINTS = (
    ("L-Elb", LEFT_ELBOW_JOINT, RIGHT_ELBOW_JOINT, True),
    ("R-Elb", RIGHT_ELBOW_JOINT, LEFT_ELBOW_JOINT, False),
    ("L-Hip", LEFT_HIP_JOINT, RIGHT_HIP_JOINT, True),
    ("R-Hip", RIGHT_HIP_JOINT, LEFT_HIP_JOINT, False),
    ("L-Knee", LEFT_KNEE_JOINT, RIGHT_KNEE_JOINT, True),
    ("R-Knee", RIGHT_KNEE_JOINT, LEFT_KNEE_JOINT, False),
)


def format_vis(value) -> str:
    """Format a landmark visibility score cleanly for the debug overlay."""
    return f"{value:.2f}" if value is not None else "N/A"


def boxes_overlap(box_a, box_b) -> bool:
    """Return True if two (x1, y1, x2, y2) rectangles overlap."""
    return not (
        box_a[2] <= box_b[0]
        or box_a[0] >= box_b[2]
        or box_a[3] <= box_b[1]
        or box_a[1] >= box_b[3]
    )


def draw_angle_label(
    frame,
    text: str,
    vertex_px,
    is_reliable: bool,
    align_right: bool,
    placed_boxes,
) -> None:
    """
    Draw outlined angle text beside a joint vertex while preventing labels
    from going outside the frame or overlapping previously placed labels.
    """
    height, width = frame.shape[:2]
    vx, vy = vertex_px

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2
    outline_thickness = 4
    margin = 10
    joint_gap = 12
    box_pad = 4

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    min_x = margin
    max_x = max(margin, width - text_w - margin)
    min_y = text_h + margin
    max_y = max(min_y, height - baseline - margin)

    def make_box(tx: int, ty: int):
        return (tx - box_pad, ty - text_h - box_pad, tx + text_w + box_pad, ty + baseline + box_pad)

    # Prefer the outer side of the joint first, then fall back to the opposite side.
    x_candidates = (
        (vx + joint_gap, vx - text_w - joint_gap)
        if align_right
        else (vx - text_w - joint_gap, vx + joint_gap)
    )
    y_step = text_h + baseline + 10
    y_offsets = (0, y_step, -y_step, 2 * y_step, -2 * y_step, 3 * y_step, -3 * y_step)

    chosen_x = max(min_x, min(x_candidates[0], max_x))
    chosen_y = max(min_y, min(vy - 8, max_y))
    chosen_box = make_box(chosen_x, chosen_y)

    found_free_spot = False
    for raw_x in x_candidates:
        cand_x = max(min_x, min(raw_x, max_x))
        for dy in y_offsets:
            cand_y = max(min_y, min(vy - 8 + dy, max_y))
            cand_box = make_box(cand_x, cand_y)
            if not any(boxes_overlap(cand_box, other) for other in placed_boxes):
                chosen_x, chosen_y, chosen_box = cand_x, cand_y, cand_box
                found_free_spot = True
                break
        if found_free_spot:
            break

    placed_boxes.append(chosen_box)
    text_color = (255, 255, 255) if is_reliable else (0, 140, 255)

    # Draw a black outline behind text for readability on any background.
    cv2.putText(
        frame,
        text,
        (chosen_x, chosen_y),
        font,
        font_scale,
        (0, 0, 0),
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (chosen_x, chosen_y),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )


def format_joint_pair_status(group_name: str, left_key: str, right_key: str, joint_results):
    """Build a compact (text, bgr_color) line for a Left/Right joint pair."""
    l_angle, l_rel = joint_results[left_key]
    r_angle, r_rel = joint_results[right_key]

    def fmt_one(side: str, angle, rel) -> str:
        if rel.is_reliable and angle is not None:
            return f"{side}:{int(round(angle))}deg"
        reason = rel.reasons[0] if rel.reasons else "hidden"
        return f"{side}:HIDDEN({reason})"

    both_ok = (l_rel.is_reliable and l_angle is not None) and (
        r_rel.is_reliable and r_angle is not None
    )
    either_ok = (l_rel.is_reliable and l_angle is not None) or (
        r_rel.is_reliable and r_angle is not None
    )
    color = (0, 255, 0) if both_ok else ((0, 200, 255) if either_ok else (0, 80, 255))
    return (f"{group_name}: {fmt_one('L', l_angle, l_rel)} | {fmt_one('R', r_angle, r_rel)}", color)


def draw_debug_panel(frame, pose, joint_results) -> None:
    """Draw a semi-transparent HUD showing visibilities and joint reliability."""
    l_sh, l_el, l_wr = get_joint_visibilities(pose, LEFT_ELBOW_JOINT)
    r_sh, r_el, r_wr = get_joint_visibilities(pose, RIGHT_ELBOW_JOINT)
    l_hip, l_kn, l_ank = get_joint_visibilities(pose, LEFT_KNEE_JOINT)
    r_hip, r_kn, r_ank = get_joint_visibilities(pose, RIGHT_KNEE_JOINT)

    lines = [
        ("DEBUG [D: toggle | R: reset reps]  L=Blue  R=Orange  Hollow=Vis<0.5", (230, 230, 230)),
        (
            f"L Vis: Sh={format_vis(l_sh)} El={format_vis(l_el)} Wr={format_vis(l_wr)} | Hip={format_vis(l_hip)} Kn={format_vis(l_kn)} Ank={format_vis(l_ank)}",
            COLOR_LEFT,
        ),
        (
            f"R Vis: Sh={format_vis(r_sh)} El={format_vis(r_el)} Wr={format_vis(r_wr)} | Hip={format_vis(r_hip)} Kn={format_vis(r_kn)} Ank={format_vis(r_ank)}",
            COLOR_RIGHT,
        ),
        format_joint_pair_status("Elbows", "L-Elb", "R-Elb", joint_results),
        format_joint_pair_status("Hips  ", "L-Hip", "R-Hip", joint_results),
        format_joint_pair_status("Knees ", "L-Knee", "R-Knee", joint_results),
    ]

    x0, y0 = 15, 55
    panel_w = min(frame.shape[1] - 30, 560)
    panel_h = 18 + len(lines) * 20

    # Semi-transparent dark backdrop so debug text stays legible.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x0 + 8, y0 + 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )


SQUAT_STAGE_COLORS = {
    SquatStage.STANDING: (0, 220, 0),
    SquatStage.DESCENDING: (0, 200, 255),
    SquatStage.BOTTOM: (255, 220, 0),
    SquatStage.ASCENDING: (120, 255, 100),
    SquatStage.UNKNOWN: (180, 180, 180),
}


def draw_squat_dashboard(frame, squat_result, counter_result) -> None:
    """Display current squat movement stage and repetition count in the top-right."""
    height, width = frame.shape[:2]
    stage_text = f"Squat: {squat_result.stage.value}"
    reps_text = f"Reps: {counter_result.reps}"
    hint_text = "[R] Reset"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (stage_w, stage_h), _ = cv2.getTextSize(stage_text, font, 0.68, 2)
    (reps_w, reps_h), _ = cv2.getTextSize(reps_text, font, 0.82, 2)
    (hint_w, hint_h), _ = cv2.getTextSize(hint_text, font, 0.38, 1)

    max_text_w = max(stage_w, reps_w + hint_w + 16)
    pad_x, pad_y = 14, 10
    box_w = max_text_w + 2 * pad_x
    box_h = stage_h + reps_h + 3 * pad_y + 4

    x1 = max(15, width - box_w - 15)
    y1 = 12
    x2 = min(width - 10, x1 + box_w)
    y2 = min(height - 10, y1 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    stage_color = SQUAT_STAGE_COLORS.get(squat_result.stage, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), stage_color, 2)

    # Line 1: Squat Stage
    cv2.putText(
        frame,
        stage_text,
        (x1 + pad_x, y1 + pad_y + stage_h),
        font,
        0.68,
        stage_color,
        2,
        cv2.LINE_AA,
    )

    # Line 2: Reps count (prominent) + Reset hint
    reps_baseline_y = y1 + pad_y + stage_h + pad_y + reps_h + 2
    cv2.putText(
        frame,
        reps_text,
        (x1 + pad_x, reps_baseline_y),
        font,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        hint_text,
        (x1 + pad_x + reps_w + 16, reps_baseline_y - 2),
        font,
        0.38,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

# ---------------------------------------------------------------------------
# 1. Load the MediaPipe Pose Landmarker Model
# ---------------------------------------------------------------------------
# Resolve the `.task` model bundle relative to this file:
#   __file__              -> .../Demo/doc/main.py
#   .parent               -> .../Demo/doc
#   .parent.parent        -> .../Demo
# This ensures the model is found no matter which folder you run the script from.
model_path = Path(__file__).resolve().parent.parent / "pose_landmarker_full.task"

# BaseOptions tells MediaPipe where the pre-trained `.task` model file lives.
base_options = python.BaseOptions(
    model_asset_path=str(model_path)
)

# PoseLandmarkerOptions configures how detection and tracking behave:
# - running_mode=VIDEO: optimized for sequential video frames; uses temporal
#   tracking between frames and requires a strictly increasing timestamp (ms).
#   (Other modes: IMAGE for single photos, LIVE_STREAM for async callbacks.)
# - num_poses: maximum number of people to detect in each frame.
# - min_pose_detection_confidence: minimum score to initially detect a person.
# - min_pose_presence_confidence: minimum score to confirm a pose is still present.
# - min_tracking_confidence: minimum score to keep tracking landmarks across frames
#   without re-running full detection.
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Instantiate the landmarker engine with the options above.
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# 2. Skeleton Connections (33 MediaPipe Pose Landmarks, indexed 0..32)
# ---------------------------------------------------------------------------
# MediaPipe returns 33 body keypoints per person (0=nose, 11/12=shoulders,
# 13/14=elbows, 15/16=wrists, 23/24=hips, 25/26=knees, 27/28=ankles, etc.).
# Each (start_idx, end_idx) tuple defines a bone line to draw between two joints.
POSE_CONNECTION = [
    # Face (nose, eyes, ears, mouth)
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),

    # Torso (shoulders and hips)
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # Left arm (shoulder -> elbow -> wrist -> hand)
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),

    # Right arm (shoulder -> elbow -> wrist -> hand)
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),

    # Left leg (hip -> knee -> ankle -> foot)
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),

    # Right leg (hip -> knee -> ankle -> foot)
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]

# ---------------------------------------------------------------------------
# 3. Webcam Setup & Main Processing Loop
# ---------------------------------------------------------------------------
# Index 0 opens the default system webcam (use 1, 2, ... for external cameras).
cap = cv2.VideoCapture(0)

prev_time = 0                      # Timestamp of the previous frame (for FPS calculation)
frame_timestamp_ms = 0             # Monotonically increasing timestamp required by VIDEO mode
debug_side_view = DEBUG_SIDE_VIEW  # Live-toggleable side-view debug flag
angle_smoother = AngleSmoother(alpha=0.45)  # Lightweight per-joint EMA smoother
squat_detector = SquatStageDetector()       # Rule-based squat movement stage detector
squat_counter = SquatRepCounter()           # Sequential squat repetition counter

# Using try...finally guarantees hardware/native resources (webcam, MediaPipe,
# OpenCV windows) are always released cleanly—even if an error occurs.
try:
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        # Read one frame from the camera; `frame` is a NumPy array (H x W x 3) in BGR.
        success, frame = cap.read()
        if not success:
            break

        # Flip horizontally (code 1 = mirror along Y-axis) so movement feels natural.
        frame = cv2.flip(frame, 1)

        # OpenCV stores images in BGR channel order, while MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the NumPy RGB array in MediaPipe's Image container (SRGB = 8-bit RGB).
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires each call to receive a strictly larger integer timestamp
        # in milliseconds (+33 ms approximates 30 FPS: 1000 ms / 30 ≈ 33 ms).
        frame_timestamp_ms += 33
        result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # -------------------------------------------------------------------
        # 4. Draw Detected Landmarks & Skeleton Overlay
        # -------------------------------------------------------------------
        # frame.shape is (height, width, channels); we only need height & width.
        height, width = frame.shape[:2]
        curr_time = time.perf_counter()

        # `result.pose_landmarks` is a list of detected poses (up to `num_poses`).
        if not result.pose_landmarks:
            # Reset all smoothed values immediately when pose tracking is lost
            # so stale angles are never carried over when tracking resumes.
            angle_smoother.reset_all()
            squat_result = squat_detector.update(None, None, None, None, curr_time)
            counter_result = squat_counter.update(squat_result)
        else:
            squat_result = None
            counter_result = None
            for pose in result.pose_landmarks:
                # Draw skeleton lines connecting landmark pairs.
                for start_idx, end_idx in POSE_CONNECTION:
                    start = pose[start_idx]
                    end = pose[end_idx]

                    # Each landmark has a `visibility` confidence (0.0-1.0) estimating
                    # whether the joint is visible (not blocked or out of frame).
                    # Skip drawing the bone if either joint is likely hidden (< 0.5).
                    vis1 = getattr(start, "visibility", None)
                    vis2 = getattr(end, "visibility", None)
                    if (vis1 is not None and vis1 < 0.5) or (vis2 is not None and vis2 < 0.5):
                        continue

                    x1, y1 = int(start.x * width), int(start.y * height)
                    x2, y2 = int(end.x * width), int(end.y * height)

                    line_color = get_connection_color(start_idx, end_idx, debug_side_view)
                    cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

                # Draw each of the 33 landmarks (side-colored in debug mode, green normally).
                # In debug mode, landmarks with visibility < 0.5 are drawn hollow so
                # occluded far-side points stand out clearly from visible near-side ones.
                for idx, landmark in enumerate(pose):
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    color = get_landmark_color(idx, debug_side_view)
                    vis = getattr(landmark, "visibility", None)
                    thickness = 1 if (debug_side_view and (vis is None or vis < 0.5)) else -1
                    cv2.circle(frame, (x, y), 5, color, thickness)

                # Track placed label rectangles each frame to prevent overlaps
                # (reserve FPS, top-right squat dashboard, and optional debug HUD).
                placed_boxes = [
                    (15, 10, 155, 48),
                    (max(15, width - 290), 10, width - 10, 105),
                ]
                if debug_side_view:
                    debug_w = min(width - 30, 560)
                    debug_h = 18 + 6 * 20
                    placed_boxes.append((15, 55, 15 + debug_w, 55 + debug_h))

                # Evaluate, smooth, and display all tracked joints (elbows, hips, knees).
                # Unreliable measurements are hidden and their smoother state is reset
                # so stale values are never displayed after occlusion or tracking loss.
                joint_results = {}
                for label_prefix, joint_triplet, opposite_triplet, align_right in TRACKED_JOINTS:
                    joint_data = get_joint_angle(
                        pose,
                        joint_triplet,
                        width,
                        height,
                        min_visibility=0.5,
                    )
                    reliability = evaluate_joint_reliability(
                        pose,
                        joint_triplet,
                        opposite_triplet,
                        width,
                        height,
                        min_visibility=0.5,
                    )

                    if joint_data is not None and reliability.is_reliable:
                        raw_angle, joint_px = joint_data
                        smoothed_angle = angle_smoother.update(label_prefix, raw_angle)
                        joint_results[label_prefix] = (smoothed_angle, reliability)

                        label = f"{label_prefix}: {int(round(smoothed_angle))} deg"
                        draw_angle_label(
                            frame,
                            label,
                            joint_px,
                            True,
                            align_right,
                            placed_boxes,
                        )
                    else:
                        angle_smoother.reset(label_prefix)
                        joint_results[label_prefix] = (None, reliability)

                # Update Squat Stage Detector using ONLY reliable smoothed knee/hip angles.
                squat_result = squat_detector.update(
                    left_knee=joint_results["L-Knee"][0],
                    right_knee=joint_results["R-Knee"][0],
                    left_hip=joint_results["L-Hip"][0],
                    right_hip=joint_results["R-Hip"][0],
                    timestamp_sec=curr_time,
                )
                counter_result = squat_counter.update(squat_result)

                if debug_side_view:
                    draw_debug_panel(frame, pose, joint_results)

        if squat_result is not None and counter_result is not None:
            draw_squat_dashboard(frame, squat_result, counter_result)

        # -------------------------------------------------------------------
        # 5. Calculate & Display FPS (Frames Per Second)
        # -------------------------------------------------------------------
        # perf_counter() is a high-resolution monotonic timer ideal for measuring
        # elapsed frame time: FPS = 1 / (seconds elapsed since previous frame).
        delta = curr_time - prev_time
        fps = 1 / delta if prev_time > 0 and delta > 0 else 0
        prev_time = curr_time

        # Draw yellow text (BGR: 0, 255, 255) at pixel position (x=20, y=40).
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,           # Font scale
            (0, 255, 255), # Text color (BGR yellow)
            2,             # Line thickness
        )

        # Show the annotated frame in an OpenCV window.
        cv2.imshow("Pose Landmarker (Task API)", frame)

        # waitKey(1) refreshes the window and waits 1 ms for a key press.
        # `& 0xFF` keeps the lowest 8 bits (standard ASCII code):
        #   27 = Esc (quit), 'd'/'D' = toggle side-view debug, 'r'/'R' = reset reps.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord("d"), ord("D")):
            debug_side_view = not debug_side_view
        elif key in (ord("r"), ord("R")):
            squat_counter.reset()
finally:
    # Always free the webcam, close MediaPipe's native resources, and close windows.
    cap.release()
    pose_landmarker.close()
    cv2.destroyAllWindows()
