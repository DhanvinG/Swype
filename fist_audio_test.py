# import cv2
# import mediapipe as mp
# import pyautogui
# import speech_recognition as sr
# import numpy as np
# import time

# # Disable PyAutoGUI fail-safe feature (not recommended)
# pyautogui.FAILSAFE = False

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# # Initialize Webcam
# cap = cv2.VideoCapture(1)  # Use the external webcam

# # Initialize the recognizer
# recognizer = sr.Recognizer()

# # Function to check if a fist is made
# def is_fist_closed(landmarks):
#     return landmarks[8].y > landmarks[6].y and \
#            landmarks[12].y > landmarks[10].y and \
#            landmarks[16].y > landmarks[14].y and \
#            landmarks[20].y > landmarks[18].y

# # Variables for timing the fist gesture
# fist_start_time = None
# fist_held_duration = 3  # seconds

# # Variables for audio detection timing
# audio_detection_start_time = None
# audio_detection_timeout = 10  # seconds

# # Flag to indicate if the microphone is active
# mic_active = False

# # Flag to indicate if the fist gesture is allowed
# fist_gesture_allowed = True

# # Variable to indicate if we are in search mode
# search_mode = False

# # Coordinates of the search bar (example coordinates, adjust as needed)
# search_bar_coords = (500, 50)

# # Main loop to process webcam frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to RGB
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # Convert the frame back to BGR for display
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Check for fist gesture to activate microphone
#             if fist_gesture_allowed and is_fist_closed(hand_landmarks.landmark):
#                 if fist_start_time is None:
#                     fist_start_time = time.time()
#                 elif time.time() - fist_start_time >= fist_held_duration:
#                     if not mic_active:
#                         print("Fist detected for 3 seconds, activating microphone...")
#                         mic_active = True
#                         fist_gesture_allowed = False
#                         audio_detection_start_time = time.time()
#             else:
#                 fist_start_time = None

#     if mic_active:
#         with sr.Microphone() as source:
#             print("Listening...")
#             try:
#                 # More responsive audio detection
#                 audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=2)
#                 print("Recognizing...")
#                 text = recognizer.recognize_google(audio)
#                 print(f"Recognized text: {text}")
                
#                 if search_mode:
#                     pyautogui.typewrite(text + ' ')  # Type the recognized text at the cursor position
#                 else:
#                     if "search" in text.lower():
#                         search_mode = True
#                         print("Search mode activated")
#                         pyautogui.moveTo(search_bar_coords[0], search_bar_coords[1])  # Move cursor to the search bar
#                         pyautogui.click()  # Click to focus on the search bar

#                 audio_detection_start_time = time.time()  # Reset the audio detection timer
#             except sr.WaitTimeoutError:
#                 print("Listening timed out, no speech detected")
#             except sr.UnknownValueError:
#                 print("Google Speech Recognition could not understand audio")
#             except sr.RequestError as e:
#                 print(f"Could not request results from Google Speech Recognition service; {e}")

#         # Check if the audio detection timeout has been reached
#         if time.time() - audio_detection_start_time >= audio_detection_timeout:
#             print("No audio detected for 10 seconds, deactivating microphone...")
#             mic_active = False
#             fist_gesture_allowed = True  # Allow fist gesture detection again
#             search_mode = False  # Reset search mode

#     cv2.imshow('Hand Tracking', image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import numpy as np
import time

# Disable PyAutoGUI fail-safe feature (not recommended)
pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(1)  # Use the external webcam

# Initialize the recognizer
recognizer = sr.Recognizer()

# Function to check if a fist is made
def is_fist_closed(landmarks):
    return landmarks[8].y > landmarks[6].y and \
           landmarks[12].y > landmarks[10].y and \
           landmarks[16].y > landmarks[14].y and \
           landmarks[20].y > landmarks[18].y

# Variables for timing the fist gesture
fist_start_time = None
fist_held_duration = 3  # seconds

# Variables for audio detection timing
audio_detection_start_time = None
audio_detection_timeout = 10  # seconds

# Flag to indicate if the microphone is active
mic_active = False

# Flag to indicate if the fist gesture is allowed
fist_gesture_allowed = True

# Variable to indicate if we are in search mode
search_mode = False

# Coordinates of the search bar (example coordinates, adjust as needed)
search_bar_coords = (500, 50)

# Load the microphone icon
mic_icon_path = "path_to_your_mic_icon.png"
mic_icon = cv2.imread(mic_icon_path, cv2.IMREAD_UNCHANGED)
mic_icon = cv2.resize(mic_icon, (50, 50))  # Resize the icon to make it smaller

# Function to overlay the microphone icon
def overlay_icon(background, icon, x, y):
    icon_h, icon_w, icon_channels = icon.shape
    background_h, background_w = background.shape[:2]

    # Ensure the icon fits within the background
    if x + icon_w > background_w or y + icon_h > background_h:
        return

    # Overlay the icon
    for i in range(icon_h):
        for j in range(icon_w):
            if icon[i, j][3] != 0:  # Check the alpha channel
                background[y + i, x + j] = icon[i, j][:3]

# Main loop to process webcam frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Convert the frame back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check for fist gesture to activate microphone
            if fist_gesture_allowed and is_fist_closed(hand_landmarks.landmark):
                if fist_start_time is None:
                    fist_start_time = time.time()
                elif time.time() - fist_start_time >= fist_held_duration:
                    if not mic_active:
                        print("Fist detected for 3 seconds, activating microphone...")
                        mic_active = True
                        fist_gesture_allowed = False
                        audio_detection_start_time = time.time()
            else:
                fist_start_time = None

    if mic_active:
        # Display the microphone icon if the mic is active
        overlay_icon(image, mic_icon, image.shape[1] - mic_icon.shape[1] - 10, image.shape[0] - mic_icon.shape[0] - 10)

        with sr.Microphone() as source:
            print("Listening...")
            try:
                # More responsive audio detection
                audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=2)
                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")

                if search_mode:
                    pyautogui.typewrite(text + ' ')  # Type the recognized text at the cursor position
                else:
                    if "search" in text.lower():
                        search_mode = True
                        print("Search mode activated")
                        pyautogui.moveTo(search_bar_coords[0], search_bar_coords[1])  # Move cursor to the search bar
                        pyautogui.click()  # Click to focus on the search bar

                audio_detection_start_time = time.time()  # Reset the audio detection timer
            except sr.WaitTimeoutError:
                print("Listening timed out, no speech detected")
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

        # Check if the audio detection timeout has been reached
        if time.time() - audio_detection_start_time >= audio_detection_timeout:
            print("No audio detected for 10 seconds, deactivating microphone...")
            mic_active = False
            fist_gesture_allowed = True  # Allow fist gesture detection again
            search_mode = False  # Reset search mode

    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    # import cv2
    # import mediapipe as mp
    # import pyautogui
    # import speech_recognition as sr
    # import numpy as np
    # import time
    # from PIL import Image, ImageTk
    # import tkinter as tk

    # # Disable PyAutoGUI fail-safe feature (not recommended)
    # pyautogui.FAILSAFE = False

    # # Initialize MediaPipe Hands
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    # mp_drawing = mp.solutions.drawing_utils

    # # Initialize Webcam
    # cap = cv2.VideoCapture(1)  # Use the external webcam

    # # Initialize the recognizer
    # recognizer = sr.Recognizer()

    # # Function to check if a fist is made
    # def is_fist_closed(landmarks):
    #     return landmarks[8].y > landmarks[6].y and \
    #            landmarks[12].y > landmarks[10].y and \
    #            landmarks[16].y > landmarks[14].y and \
    #            landmarks[20].y > landmarks[18].y

    # # Variables for timing the fist gesture
    # fist_start_time = None
    # fist_held_duration = 3  # seconds

    # # Variables for audio detection timing
    # audio_detection_start_time = None
    # audio_detection_timeout = 10  # seconds

    # # Flag to indicate if the microphone is active
    # mic_active = False

    # # Flag to indicate if the fist gesture is allowed
    # fist_gesture_allowed = True

    # # Variable to indicate if we are in search mode
    # search_mode = False

    # # Coordinates of the search bar (example coordinates, adjust as needed)
    # search_bar_coords = (500, 50)

    # # Load the microphone icon
    # mic_icon_path = "path_to_your_mic_icon.png"  # Replace with the path to your mic icon image
    # mic_icon = Image.open(mic_icon_path).resize((50, 50), Image.LANCZOS)

    # # Function to show the microphone icon
    # def show_mic_icon():
    #     root = tk.Tk()
    #     root.overrideredirect(True)  # Remove window decorations
    #     root.geometry(f"50x50+{screen_width-60}+{screen_height-60}")  # Set the position to bottom-right
    #     root.wm_attributes("-topmost", True)  # Keep the window on top

    #     # Convert the image to a Tkinter-compatible format
    #     mic_icon_tk = ImageTk.PhotoImage(mic_icon)
    #     label = tk.Label(root, image=mic_icon_tk, bg='white')
    #     label.pack()

    #     return root

    # # Hide the microphone icon
    # def hide_mic_icon(root):
    #     root.destroy()

    # # Get screen dimensions
    # screen_width, screen_height = pyautogui.size()

    # # Main loop to process webcam frames
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # Convert the frame to RGB
    #     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = hands.process(image)

    #     # Convert the frame back to BGR for display
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #             # Check for fist gesture to activate microphone
    #             if fist_gesture_allowed and is_fist_closed(hand_landmarks.landmark):
    #                 if fist_start_time is None:
    #                     fist_start_time = time.time()
    #                 elif time.time() - fist_start_time >= fist_held_duration:
    #                     if not mic_active:
    #                         print("Fist detected for 3 seconds, activating microphone...")
    #                         mic_active = True
    #                         fist_gesture_allowed = False
    #                         audio_detection_start_time = time.time()
    #                         mic_window = show_mic_icon()  # Show the microphone icon
    #             else:
    #                 fist_start_time = None

    #     if mic_active:
    #         with sr.Microphone() as source:
    #             print("Listening...")
    #             try:
    #                 # More responsive audio detection
    #                 audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=2)
    #                 print("Recognizing...")
    #                 text = recognizer.recognize_google(audio)
    #                 print(f"Recognized text: {text}")
                    
    #                 if search_mode:
    #                     pyautogui.typewrite(text + ' ')  # Type the recognized text at the cursor position
    #                 else:
    #                     if "search" in text.lower():
    #                         search_mode = True
    #                         print("Search mode activated")
    #                         pyautogui.moveTo(search_bar_coords[0], search_bar_coords[1])  # Move cursor to the search bar
    #                         pyautogui.click()  # Click to focus on the search bar

    #                 audio_detection_start_time = time.time()  # Reset the audio detection timer
    #             except sr.WaitTimeoutError:
    #                 print("Listening timed out, no speech detected")
    #             except sr.UnknownValueError:
    #                 print("Google Speech Recognition could not understand audio")
    #             except sr.RequestError as e:
    #                 print(f"Could not request results from Google Speech Recognition service; {e}")

    #         # Check if the audio detection timeout has been reached
    #         if time.time() - audio_detection_start_time >= audio_detection_timeout:
    #             print("No audio detected for 10 seconds, deactivating microphone...")
    #             mic_active = False
    #             fist_gesture_allowed = True  # Allow fist gesture detection again
    #             search_mode = False  # Reset search mode
    #             hide_mic_icon(mic_window)  # Hide the microphone icon

    #     cv2.imshow('Hand Tracking', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()











