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

from angles import LEFT_ELBOW_JOINT, get_joint_angle

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

prev_time = 0           # Timestamp of the previous frame (for FPS calculation)
frame_timestamp_ms = 0  # Monotonically increasing timestamp required by VIDEO mode

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

        # `result.pose_landmarks` is a list of detected poses (up to `num_poses`).
        if result.pose_landmarks:
            for pose in result.pose_landmarks:
                # Draw a filled green dot on each of the 33 landmarks.
                for landmark in pose:
                    # Landmarks are normalized to [0.0, 1.0] relative to image size,
                    # so multiply by width/height to get pixel coordinates.
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    # Note: OpenCV colors are BGR -> (0, 255, 0) = green; -1 = filled.
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

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

                    # BGR (255, 255, 0) = cyan, line thickness = 2 px.
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Calculate and display the Left Elbow angle (landmarks 11 -> 13 -> 15)
                # only when shoulder, elbow, and wrist all meet the visibility threshold.
                elbow_data = get_joint_angle(
                    pose,
                    LEFT_ELBOW_JOINT,
                    width,
                    height,
                    min_visibility=0.5,
                )
                if elbow_data is not None:
                    left_elbow_angle, (elbow_x, elbow_y) = elbow_data
                    label = f"{int(round(left_elbow_angle))} deg"
                    text_x = max(10, min(elbow_x + 12, width - 110))
                    text_y = max(30, min(elbow_y - 10, height - 10))

                    # Draw a black outline behind white text for readability on any background.
                    cv2.putText(
                        frame,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # -------------------------------------------------------------------
        # 5. Calculate & Display FPS (Frames Per Second)
        # -------------------------------------------------------------------
        # perf_counter() is a high-resolution monotonic timer ideal for measuring
        # elapsed frame time: FPS = 1 / (seconds elapsed since previous frame).
        curr_time = time.perf_counter()
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
        # `& 0xFF` keeps the lowest 8 bits (standard ASCII code); 27 = Esc key.
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    # Always free the webcam, close MediaPipe's native resources, and close windows.
    cap.release()
    pose_landmarker.close()
    cv2.destroyAllWindows()
