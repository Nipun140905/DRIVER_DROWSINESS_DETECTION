# detection.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

from utils import preprocess_eye, preprocess_mouth

# Paths to model files
# Paths to model files (location-independent)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EYE_MODEL_PATH = os.path.join(BASE_DIR, "models", "eye_state_model.h5")
YAWN_MODEL_PATH = os.path.join(BASE_DIR, "models", "yawn_model.h5")

# Load the trained models
eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices based on MediaPipe documentation (tweak if needed)
LEFT_EYE_IDX = [33, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_IDX = [362, 398, 384, 385, 386, 387, 388, 466]
MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

def extract_roi(frame, landmarks, indices, padding=5):
    h, w, _ = frame.shape
    points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*points)
    xmin, xmax = max(min(x_coords) - padding, 0), min(max(x_coords) + padding, w)
    ymin, ymax = max(min(y_coords) - padding, 0), min(max(y_coords) + padding, h)
    roi = frame[ymin:ymax, xmin:xmax]
    bbox = (xmin, ymin, xmax, ymax)
    return roi, bbox

def predict_eye_state(frame, face_landmarks):
    """Returns True if either eye is closed, else False; also returns bounding box for drawing"""
    left_eye, left_bbox = extract_roi(frame, face_landmarks, LEFT_EYE_IDX,padding=5)
    right_eye, right_bbox = extract_roi(frame, face_landmarks, RIGHT_EYE_IDX,padding=5)
    closed = False

    for eye_img in [left_eye, right_eye]:
        if eye_img.size == 0:
            continue
        gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        pred = eye_model.predict(preprocess_eye(gray_eye), verbose=0)[0][0]
        if pred < 0.5:  # adjust threshold if needed
            closed = True
    return closed, left_bbox, right_bbox

def predict_yawn(frame, face_landmarks):
    """Returns True if yawn is detected, else False; also returns bounding box for drawing"""
    mouth, mouth_bbox = extract_roi(frame, face_landmarks, MOUTH_IDX,padding=35)
    yawned = False
    if mouth.size != 0:
        pred = yawn_model.predict(preprocess_mouth(mouth), verbose=0)[0][0]
        if pred > 0.45:  # adjust threshold if needed
            yawned = True
    return yawned, mouth_bbox

def get_face_landmarks(frame):
    """Detect face and return landmarks; else None"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None
