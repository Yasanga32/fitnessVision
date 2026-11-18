import time
import cv2
import mediapipe as mp
from meadiapip.tasks import python 
from mediapipe.tasks.python import vision

#load Pose Landmark Model
base_options = python.BaseOptions(
    model_assets_path="pose_landmarker_full.taks"
)
options = vision.PoseLandmarkerOptions(
    
)