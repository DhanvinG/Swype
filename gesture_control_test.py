import cv2 #current working
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
kf.R = np.array([[0.05, 0],  # Lower measurement noise for more precision
                 [0, 0.05]])
kf.Q = np.array([[0.001, 0, 0, 0],  # Lower process noise
                 [0, 0.001, 0, 0],
                 [0, 0, 0.001, 0],
                 [0, 0, 0, 0.001]])

# To measure the processing frame rate
prev_time = 0
fps = 120  # Target frame rate for higher responsiveness

# Initialize previous smoothed positions for interpolation
prev_smoothed_x, prev_smoothed_y = pyautogui.position()

# Scaling factor for movement amplification
base_scaling_factor = 1.6  # Adjust this factor to control sensitivity

# State variables to track the click states
left_click_pressed = False
right_click_pressed = False

# Initialize previous positions for low-pass filter
prev_filtered_x, prev_filtered_y = pyautogui.position()

# Low-pass filter factor
low_pass_factor = 0.15  # Increase smoothing

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

# Function to normalize hand position
def normalize_coordinates(x, y, image_width, image_height):
    norm_x = x / image_width
    norm_y = y / image_height
    return norm_x, norm_y

# Function to calculate dynamic scaling factor based on velocity
def calculate_scaling_factor(velocity, base_factor=2.1):
    # Scale factor inversely proportional to velocity for smoother movements
    dynamic_factor = base_factor * (1 / (1 + velocity))
    return dynamic_factor

# Function to calculate velocity
def calculate_velocity(prev_x, prev_y, curr_x, curr_y, elapsed_time):
    distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
    velocity = distance / elapsed_time if elapsed_time > 0 else 0
    return velocity

# Initialize previous coordinates and time for velocity calculation
prev_x, prev_y = prev_smoothed_x, prev_smoothed_y
prev_time_for_velocity = time.time()

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

            # Normalize the coordinates
            norm_x, norm_y = normalize_coordinates(target_x, target_y, screen_width, screen_height)

            # Calculate velocity
            curr_time_for_velocity = time.time()
            velocity = calculate_velocity(prev_x, prev_y, norm_x, norm_y, curr_time_for_velocity - prev_time_for_velocity)
            prev_x, prev_y = norm_x, norm_y
            prev_time_for_velocity = curr_time_for_velocity

            # Calculate dynamic scaling factor based on velocity
            dynamic_scaling_factor = calculate_scaling_factor(velocity, base_scaling_factor)

            # Apply dynamic scaling to the movements
            scaled_x = target_x * dynamic_scaling_factor
            scaled_y = target_y * dynamic_scaling_factor

            # Update the Kalman filter with the new measurements
            kf.predict()
            kf.update([scaled_x, scaled_y])

            # Get the filtered coordinates
            smoothed_x, smoothed_y = kf.x[:2]

            # Exponential moving average for smoother transition
            alpha = 0.6  # Higher smoothing factor, can be adjusted for more smoothness
            final_x = alpha * smoothed_x + (1 - alpha) * prev_smoothed_x
            final_y = alpha * smoothed_y + (1 - alpha) * prev_smoothed_y

            # Apply the low-pass filter
            filtered_x = low_pass_factor * final_x + (1 - low_pass_factor) * prev_filtered_x
            filtered_y = low_pass_factor * final_y + (1 - low_pass_factor) * prev_filtered_y

            # Move the cursor to the filtered position
            pyautogui.moveTo(int(filtered_x), int(filtered_y))

            # Update previous smoothed positions
            prev_smoothed_x, prev_smoothed_y = final_x, final_y
            prev_filtered_x, prev_filtered_y = filtered_x, filtered_y

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
