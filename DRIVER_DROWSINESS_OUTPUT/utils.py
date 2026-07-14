import cv2
import numpy as np
import time
from config import COLOR_NORMAL, COLOR_FATIGUE, COLOR_DROWSY, FONT, FONT_SCALE, THICKNESS

# FPS tracking
_prev_time = time.time()

def preprocess_eye(eye_img):
    """Resize and normalize eye image for model prediction"""
    img = cv2.resize(eye_img, (24, 24))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img
 
def preprocess_mouth(mouth_img):
    rgb_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb_img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
 
def get_fps():
    global _prev_time
    curr_time = time.time()
    fps = 1 / (curr_time - _prev_time + 1e-6)
    _prev_time = curr_time
    return int(fps)
 
def draw_status(frame, status, score):
    height, width = frame.shape[:2]
    MAX_SCORE = 40
 
    # --- Color based on status ---
    if status == "Normal":
        accent = COLOR_NORMAL       # green
    elif status == "Fatigue":
        accent = COLOR_FATIGUE      # yellow
    else:
        accent = COLOR_DROWSY       # red
 
    # ── BOTTOM HUD PANEL ──────────────────────────────────────────
    panel_h = 70
    panel_y = height - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (width, height), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
 
    # Thin accent line at top of panel
    cv2.line(frame, (0, panel_y), (width, panel_y), accent, 2)
 
    font = cv2.FONT_HERSHEY_DUPLEX
    font_sm = cv2.FONT_HERSHEY_SIMPLEX
    text_y = panel_y + 44
 
    # ── STATUS with pulsing indicator dot ─────────────────────────
    dot_x, dot_y = 22, panel_y + 35
    cv2.circle(frame, (dot_x, dot_y), 8, accent, -1)
    cv2.circle(frame, (dot_x, dot_y), 8, (255, 255, 255), 1)   # white ring
 
    cv2.putText(frame, status.upper(), (40, text_y),
                font, 0.95, accent, 2)
 
    # ── SCORE PROGRESS BAR ────────────────────────────────────────
    bar_x      = width // 2 - 80
    bar_w      = 200
    bar_h      = 10
    bar_y      = panel_y + 28
    fill_w     = int(bar_w * min(score, MAX_SCORE) / MAX_SCORE)
 
    # Background track
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), cv2.FILLED)
    # Filled portion
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      accent, cv2.FILLED)
    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (120, 120, 120), 1)
 
    # Score label above bar
    cv2.putText(frame, f"SCORE  {score}/{MAX_SCORE}",
                (bar_x, bar_y - 6), font_sm, 0.45, (200, 200, 200), 1)
 
    # ── FPS (top-right) ───────────────────────────────────────────
    fps = get_fps()
    fps_text = f"FPS  {fps}"
    (tw, _), _ = cv2.getTextSize(fps_text, font_sm, 0.5, 1)
    cv2.putText(frame, fps_text, (width - tw - 14, 26),
                font_sm, 0.5, (180, 180, 180), 1)
 
    # ── TOP-LEFT watermark ────────────────────────────────────────
    cv2.putText(frame, "DROWSINESS MONITOR", (12, 26),
                font_sm, 0.45, (100, 100, 100), 1)
 
 
def draw_box(frame, bbox, color=(255, 255, 255), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
