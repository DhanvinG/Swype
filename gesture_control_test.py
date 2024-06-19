# import pyautogui

# # Perform a right-click action
# pyautogui.mouseDown(button='right')
# pyautogui.mouseUp(button='right')

import cv2 #Smooth gesture control (attempting to implement)
import mediapipe as mp
import pyautogui
from filterpy.kalman import KalmanFilter
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

screen_width, screen_height = pyautogui.size()
frame_count = 0
n = 1  # Process every frame for smoother movement

# Initialize Kalman filter for smoothing
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.array([0, 0, 0, 0])  # initial state (location and velocity)
kf.F = np.array([[1, 0, 1, 0],  # state transition matrix
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],  # measurement function
                    [0, 1, 0, 0]])
kf.P *= 1000  # covariance matrix
kf.R = np.array([[0.3, 0],  # measurement noise
                    [0, 0.3]])
kf.Q = np.array([[0.005, 0, 0, 0],  # process noise
                    [0, 0.005, 0, 0],
                    [0, 0, 0.005, 0],
                    [0, 0, 0, 0.005]])

# To measure the processing frame rate
prev_time = 0
fps = 60  # Target frame rate for higher responsiveness

# Initialize previous smoothed positions for interpolation
prev_smoothed_x, prev_smoothed_y = pyautogui.position()

# Scaling factor for movement amplification
scaling_factor = 2.1  # Adjust this factor to control sensitivity

# State variables to track the click states
left_click_pressed = False
right_click_pressed = False

# Function to check if the thumb and index finger are touching
def is_thumb_index_touching(landmarks, threshold=0.099):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    return distance < threshold  # Adjust the threshold as needed

# Function to check if the thumb and middle finger are touching
def is_thumb_middle_touching(landmarks, threshold=0.099):
    thumb_tip = landmarks[4]
    middle_tip = landmarks[12]
    distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y]))
    return distance < threshold  # Adjust the threshold as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Limit the frame rate to the target fps
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time < 1.0 / fps:
        continue
    prev_time = current_time

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Convert the frame back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            target_x = int((1 - index_finger_tip.x) * screen_width)  # Invert the X coordinate
            target_y = int(index_finger_tip.y * screen_height)

            # Apply scaling to the movements
            scaled_x = target_x * scaling_factor
            scaled_y = target_y * scaling_factor

            # Update the Kalman filter with the new measurements
            kf.predict()
            kf.update([scaled_x, scaled_y])

            # Get the filtered coordinates
            smoothed_x, smoothed_y = kf.x[:2]

            # Exponential moving average for smoother transition
            alpha = 0.4  # Smoothing factor, can be adjusted for smoothness
            final_x = alpha * smoothed_x + (1 - alpha) * prev_smoothed_x
            final_y = alpha * smoothed_y + (1 - alpha) * prev_smoothed_y

            # Move the cursor to the interpolated position
            pyautogui.moveTo(int(final_x), int(final_y))

            # Update previous smoothed positions
            prev_smoothed_x, prev_smoothed_y = final_x, final_y

            # Check for left-click gesture (thumb and index finger touching)
            if is_thumb_index_touching(hand_landmarks.landmark):
                if not left_click_pressed:
                    pyautogui.mouseDown(button='left')
                    left_click_pressed = True
                    print("Left-click activated")
            else:
                if left_click_pressed:
                    pyautogui.mouseUp(button='left')
                    left_click_pressed = False
                    print("Left-click deactivated")

            # Check for right-click gesture (thumb and middle finger touching)
            if is_thumb_middle_touching(hand_landmarks.landmark):
                if not right_click_pressed:
                    pyautogui.mouseDown(button='right')
                    right_click_pressed = True
                    print("Right-click activated")
            else:
                if right_click_pressed:
                    pyautogui.mouseUp(button='right')
                    right_click_pressed = False
                    print("Right-click deactivated")

    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

