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

        cv2.imshow("Pose Landmarker (Task API)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    pose_landmarker.close()
    cv2.destroyAllWindows()

