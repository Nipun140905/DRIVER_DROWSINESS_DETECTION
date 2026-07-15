# Driver Drowsiness Detection System 🚗💤

A real-time driver drowsiness detection system using **deep learning and computer vision**.  
The system detects **eye closure** and **yawning**, computes a drowsiness score, and triggers an **audio alarm** when the driver is detected as drowsy.

---

## Project Overview
Driver drowsiness is a major cause of road accidents.  
This project addresses the problem by:
- Training two CNN-based models:
  - Eye state detection
  - Yawn detection
- Integrating both models using **OpenCV** for real-time webcam monitoring
- Using a **scoring logic** to classify driver state as:
  - Normal
  - Fatigue
  - Drowsy
- Triggering an alarm when the drowsiness threshold is crossed

---

## Models Used
- **Eye State Detection Model** – detects whether eyes are open or closed
- **Yawn Detection Model** – detects yawning from mouth region

Both models are trained separately and saved as `.h5` files, then loaded into the real-time application.

---

## Project Structure
```text
driver-drowsiness-detection/
│
├── DRIVER_DROWSINESS_EYE_GIT.ipynb
├── DRIVER_DROWSINESS_YAWN.ipynb
│
├── DRIVER_DROWSINESS_OUTPUT/
│ ├── models/
│ │ ├── eye_state_model.keras
│ │ ├── face_landmarker.task
│ │ ├── yawn_model.keras
│ ├── main.py
│ ├── detection.py
│ ├── utils.py
│ ├── config.py
│ ├── alarm.py
│ ├── alarm.mp3
│ ├── requirements.txt
│
├── README.md
├── .gitignore

```
---

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- playsound

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/driver-drowsiness-detection.git


Install dependencies:

pip install -r DRIVER_DROWSINESS_OUTPUT/requirements.txt


Run the application:
python DRIVER_DROWSINESS_OUTPUT/main.py
A webcam is required to run the system.

Results

The system successfully detects eye closure and yawning in real time

Drowsiness score increases based on detected events

Alarm is triggered when drowsiness exceeds the defined threshold

Output videos are not included in this repository due to privacy concerns.

Contributions

This is a group project.

Eye Drowsiness Detection Model: Developed by Prashant Bansal

Yawn Detection Model, System Integration, OpenCV Pipeline, Scoring Logic:
Developed by Nipun Garg

Notes

The trained models are included for demonstration purposes

The notebooks contain the complete training and evaluation workflow

This project is intended for academic and learning purposes


---

