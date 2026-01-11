# alarm.py
import threading
from playsound import playsound
from config import ALARM_FILE

alarm_playing = False
continuous_thread = None

def play_alarm_continuous():
    global alarm_playing
    while alarm_playing:
        playsound('ALARM_FILE')

def start_alarm():
    global alarm_playing, continuous_thread
    if not alarm_playing:
        alarm_playing = True
        continuous_thread = threading.Thread(target=play_alarm_continuous, daemon=True)
        continuous_thread.start()

def stop_alarm():
    global alarm_playing
    alarm_playing = False

