from pathlib import Path
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load Pose Landmark Model
model_path = Path(__file__).resolve().parent.parent / "pose_landmarker_full.task"

base_options = python.BaseOptions(
    model_asset_path=str(model_path)
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Manual pose connections (33 landmarks)
POSE_CONNECTION = [
    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),

    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # Left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),

    # Right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),

    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),

    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]

# Webcam setup
cap = cv2.VideoCapture(0)

# Main processing loop
prev_time = 0
frame_timestamp_ms = 0

try:
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_timestamp_ms += 33
        result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw circles on landmarks
        height, width = frame.shape[:2]
        if result.pose_landmarks:
            for pose in result.pose_landmarks:
                for landmark in pose:
                    x = int(landmark.x * width)  # convert normalized x to pixel coordinate
                    y = int(landmark.y * height)  # convert normalized y to pixel coordinate
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw skeleton connections
                for start_idx, end_idx in POSE_CONNECTION:
                    start = pose[start_idx]
                    end = pose[end_idx]
                    vis1 = getattr(start, "visibility", None)
                    vis2 = getattr(end, "visibility", None)
                    if (vis1 is not None and vis1 < 0.5) or (vis2 is not None and vis2 < 0.5):
                        continue
                    x1, y1 = int(start.x * width), int(start.y * height)
                    x2, y2 = int(end.x * width), int(end.y * height)

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Calculate FPS
        curr_time = time.perf_counter()
        delta = curr_time - prev_time
        fps = 1 / delta if prev_time > 0 and delta > 0 else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Pose Landmarker (Task API)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    pose_landmarker.close()
    cv2.destroyAllWindows()

