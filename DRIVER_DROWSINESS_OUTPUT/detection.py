# detection.py

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import os

from utils import preprocess_eye, preprocess_mouth
 
# Paths to model files (location-independent)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EYE_MODEL_PATH  = os.path.join(BASE_DIR, "models", "eye_state_model.keras")
YAWN_MODEL_PATH = os.path.join(BASE_DIR, "models", "yawn_model.keras")
FACE_LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
 
# Load the trained models
eye_model  = tf.keras.models.load_model(EYE_MODEL_PATH)
yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)

# MediaPipe FaceLandmarker setup (new API, works on 0.10.x+)
base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)
 
# Landmark indices based on MediaPipe documentation
LEFT_EYE_IDX  = [33, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_IDX = [362, 398, 384, 385, 386, 387, 388, 466]
MOUTH_IDX     = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
 
def extract_roi(frame, landmarks, indices, padding=5):
    h, w, _ = frame.shape
    # New API: landmarks is a list of NormalizedLandmark objects (no .landmark sub-list)
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*points)
    xmin = max(min(x_coords) - padding, 0)
    xmax = min(max(x_coords) + padding, w)
    ymin = max(min(y_coords) - padding, 0)
    ymax = min(max(y_coords) + padding, h)
    roi  = frame[ymin:ymax, xmin:xmax]
    bbox = (xmin, ymin, xmax, ymax)
    return roi, bbox
 
def predict_eye_state(frame, face_landmarks):
    """Returns True if either eye is closed; also returns bounding boxes."""
    left_eye,  left_bbox  = extract_roi(frame, face_landmarks, LEFT_EYE_IDX,  padding=5)
    right_eye, right_bbox = extract_roi(frame, face_landmarks, RIGHT_EYE_IDX, padding=5)
    closed = False
 
    for eye_img in [left_eye, right_eye]:
        if eye_img.size == 0:
            continue
        gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        pred = eye_model.predict(preprocess_eye(gray_eye), verbose=0)[0][0]
        if pred < 0.5:
            closed = True
 
    return closed, left_bbox, right_bbox
 
def predict_yawn(frame, face_landmarks):
    """Returns True if yawn detected; also returns bounding box."""
    mouth, mouth_bbox = extract_roi(frame, face_landmarks, MOUTH_IDX, padding=35)
    yawned = False
    if mouth.size != 0:
        pred = yawn_model.predict(preprocess_mouth(mouth), verbose=0)[0][0]
        if pred > 0.45:
            yawned = True
    return yawned, mouth_bbox
 
def get_face_landmarks(frame):
    """Detect face and return landmarks list; returns None if no face found."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)
    if result.face_landmarks:
        return result.face_landmarks[0]  # list of NormalizedLandmark
    return None