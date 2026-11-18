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
    base_options=base_options
)
