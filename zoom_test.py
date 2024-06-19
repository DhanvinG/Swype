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


# import cv2
# import mediapipe as mp
# import pyautogui
# import numpy as np

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# # Initialize Webcam
# cap = cv2.VideoCapture(1)  # Use the external webcam

# # Function to calculate the distance between two points
# def calculate_distance(point1, point2):
#     return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

# # Variables to store previous distances for zooming
# prev_distance = None
# frame_count = 0  # Frame counter
# frame_skip = 10  # Check every 10 frames

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Increment the frame counter
#     frame_count += 1

#     # Skip frames
#     if frame_count % frame_skip != 0:
#         continue

#     # Convert the frame to RGB
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # Convert the frame back to BGR for display
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     if results.multi_hand_landmarks:
#         if len(results.multi_hand_landmarks) == 2:
#             right_hand = results.multi_hand_landmarks[0]
#             left_hand = results.multi_hand_landmarks[1]

#             # Calculate the distance between the thumb tips of both hands
#             right_thumb_tip = right_hand.landmark[4]
#             left_thumb_tip = left_hand.landmark[4]

#             distance = calculate_distance(right_thumb_tip, left_thumb_tip)

#             if prev_distance is not None:
#                 zoom_threshold = 1.05  # Threshold factor for more stability
#                 # Zoom out if the distance increases significantly
#                 if distance >= prev_distance * zoom_threshold:
#                     pyautogui.hotkey('ctrl', '-')
#                     print("Zooming out")
#                 # Zoom in if the distance decreases significantly
#                 elif distance <= prev_distance / zoom_threshold:
#                     pyautogui.hotkey('ctrl', '+')
#                     print("Zooming in")

#             prev_distance = distance

#             mp_drawing.draw_landmarks(image, right_hand, mp_hands.HAND_CONNECTIONS)
#             mp_drawing.draw_landmarks(image, left_hand, mp_hands.HAND_CONNECTIONS)

#     cv2.imshow('Hand Tracking', image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import pyautogui
# import numpy as np
# from collections import deque

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# # Initialize Webcam
# cap = cv2.VideoCapture(1)  # Use the external webcam

# # Function to check if thumb and index finger are touching
# def is_thumb_index_touching(landmarks, threshold=0.05):
#     thumb_tip = landmarks[4]
#     index_tip = landmarks[8]
#     distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
#     return distance < threshold  # Adjust the threshold as needed

# # Function to calculate the distance between two points
# def calculate_distance(point1, point2):
#     return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

# # Buffer for storing recent distances
# distance_buffer = deque(maxlen=10)

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
#         if len(results.multi_hand_landmarks) == 2:
#             right_hand = results.multi_hand_landmarks[0]
#             left_hand = results.multi_hand_landmarks[1]

#             # Check if thumbs and index fingers of both hands are touching
#             if is_thumb_index_touching(right_hand.landmark) and is_thumb_index_touching(left_hand.landmark):
#                 # Calculate the distance between the thumbs and index fingers of both hands
#                 right_distance = calculate_distance(right_hand.landmark[4], right_hand.landmark[8])
#                 left_distance = calculate_distance(left_hand.landmark[4], left_hand.landmark[8])

#                 avg_distance = (right_distance + left_distance) / 2
#                 distance_buffer.append(avg_distance)

#                 if len(distance_buffer) == distance_buffer.maxlen:
#                     avg_distance = np.mean(distance_buffer)
#                     prev_avg_distance = np.mean(list(distance_buffer)[:-1])

#                     zoom_factor = 1.1

#                     # Zoom out if the distance increases significantly
#                     if avg_distance >= prev_avg_distance * zoom_factor:
#                         pyautogui.hotkey('ctrl', '-')
#                         print("Zooming out")
#                     # Zoom in if the distance decreases significantly
#                     elif avg_distance <= prev_avg_distance / zoom_factor:
#                         pyautogui.hotkey('ctrl', '+')
#                         print("Zooming in")

#             mp_drawing.draw_landmarks(image, right_hand, mp_hands.HAND_CONNECTIONS)
#             mp_drawing.draw_landmarks(image, left_hand, mp_hands.HAND_CONNECTIONS)

#     cv2.imshow('Hand Tracking', image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

