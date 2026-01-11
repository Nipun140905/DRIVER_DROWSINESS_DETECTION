# config.py
import cv2
# Scoring system
EYE_CLOSED_SCORE = 2
YAWN_SCORE = 5
NORMALIZE_SCORE = -1

# Status thresholds
NORMAL_MAX = 10
FATIGUE_MAX = 20

# UI Colors (BGR format for OpenCV)
COLOR_NORMAL = (0, 255, 0)      # Green
COLOR_FATIGUE = (0, 255, 255)   # Yellow
COLOR_DROWSY = (0, 0, 255)      # Red
COLOR_BG = (30, 30, 30)         # Dark grey for backgrounds

# Fonts
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
THICKNESS = 3

# Alarm
ALARM_FILE = 'alarm.mp3'
