# main.py

import cv2
from config import EYE_CLOSED_SCORE, YAWN_SCORE, NORMALIZE_SCORE, NORMAL_MAX, FATIGUE_MAX
from config import COLOR_NORMAL, COLOR_FATIGUE, COLOR_DROWSY
from utils import draw_status, draw_box
from detection import get_face_landmarks, predict_eye_state, predict_yawn
from alarm import start_alarm, stop_alarm

def get_status_and_color(score):
    if score < NORMAL_MAX:
        return "Normal", COLOR_NORMAL
    elif score < FATIGUE_MAX:
        return "Fatigue", COLOR_FATIGUE
    else:
        return "Drowsy", COLOR_DROWSY

def clamp_score(score):
    return max(0, min(score, 40))  # Prevent surprisingly high or negative values

def main():
    window_name = "Driver Drowsiness Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480) 
    cap = cv2.VideoCapture(0)
    score = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_landmarks = get_face_landmarks(frame)
        eye_closed = False
        yawned = False
        left_bbox = right_bbox = mouth_bbox = None

        if face_landmarks:
            # EYE DETECTION
            eye_closed, left_bbox, right_bbox = predict_eye_state(frame, face_landmarks)
            # YAWN DETECTION
            yawned, mouth_bbox = predict_yawn(frame, face_landmarks)

        # Update score per your rules
        if eye_closed and yawned:
            score += EYE_CLOSED_SCORE + YAWN_SCORE
        elif eye_closed:
            score += EYE_CLOSED_SCORE
        elif yawned:
            score += YAWN_SCORE
        else:
            score += NORMALIZE_SCORE

        score = clamp_score(score)

        # Get status and color
        status, box_color = get_status_and_color(score)

        
        # Play alarm if drowsy
        if status == "Drowsy":
            start_alarm()
        else:
            stop_alarm()
        draw_status(frame, status, score)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit if ESC is pressed or window is closed
        # cv2.getWindowProperty returns < 0 if window is closed
        if key == 27 or cv2.getWindowProperty("Driver Drowsiness Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
