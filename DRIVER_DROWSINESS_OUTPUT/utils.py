# utils.py

import cv2
import numpy as np
from config import COLOR_NORMAL, COLOR_FATIGUE, COLOR_DROWSY, FONT, FONT_SCALE, THICKNESS

def preprocess_eye(eye_img):
    """Resize and normalize eye image for model prediction"""
    img = cv2.resize(eye_img, (24, 24))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def preprocess_mouth(mouth_img):
    rgb_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB)  # Convert the mouth image from BGR to RGB
    img = cv2.resize(rgb_img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def draw_status(frame, status, score):
    # Banner with color based on alert level (green=Normal, yellow=Fatigue, red=Drowsy)
    import cv2
    from config import COLOR_NORMAL, COLOR_FATIGUE, COLOR_DROWSY
    height, width = frame.shape[:2]
    overlay = frame.copy()
    rect_height = 60
    y_pos = height - rect_height - 10

    # Dynamically select color by current status text
    if status == "Normal":
        banner_color = COLOR_NORMAL
    elif status == "Fatigue":
        banner_color = COLOR_FATIGUE
    else:
        banner_color = COLOR_DROWSY

    cv2.rectangle(overlay, (0, y_pos), (width, y_pos + rect_height), banner_color, cv2.FILLED)
    alpha = 0.5  # Professional semi-transparent look
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness = 2
    cv2.putText(frame, f"Status: {status}", (30, y_pos + 38), font, font_scale, (255,255,255), thickness)
    cv2.putText(frame, f"Score: {score}", (width - 220, y_pos + 38), font, font_scale, (255,255,255), thickness)


def draw_box(frame, bbox, color=(255, 255, 255), thickness=2):
    """Draws a rectangle (bounding box) on frame"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
