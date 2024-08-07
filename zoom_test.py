import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(1)  # Use the external webcam

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

# Variables to store previous distances for zooming
prev_distance = None
frame_count = 0  # Frame counter
frame_skip = 10  # Check every 10 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increment the frame counter
    frame_count += 1

    # Skip frames
    if frame_count % frame_skip != 0:
        continue

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Convert the frame back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            right_hand = results.multi_hand_landmarks[0]
            left_hand = results.multi_hand_landmarks[1]

            # Calculate the distance between the thumb tips of both hands
            right_thumb_tip = right_hand.landmark[4]
            left_thumb_tip = left_hand.landmark[4]

            distance = calculate_distance(right_thumb_tip, left_thumb_tip)

            if prev_distance is not None:
                zoom_threshold = 1.05  # Threshold factor for more stability
                # Zoom out if the distance increases significantly
                if distance >= prev_distance * zoom_threshold:
                    pyautogui.hotkey('ctrl', '-')
                    print("Zooming out")
                # Zoom in if the distance decreases significantly
                elif distance <= prev_distance / zoom_threshold:
                    pyautogui.hotkey('ctrl', '+')
                    print("Zooming in")

            prev_distance = distance

            mp_drawing.draw_landmarks(image, right_hand, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, left_hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
