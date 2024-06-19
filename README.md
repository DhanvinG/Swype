GitHub Repository: Gesture and Voice Control Systems
Welcome to the GitHub repository for our gesture and voice control projects. This repository contains the following main files and their descriptions, along with the necessary pip install commands to get you started.

Files and Descriptions
fist_audio_test.py
Description: This script allows you to activate audio by making a fist with your hand for 3 seconds. Once activated, it continues to analyze speech and convert it to text until no audio is registered for 10 seconds. Additionally, saying "Search" will automatically navigate to the search bar.

gesture_control_test.py
Description: This script enables cursor control using hand gestures. The cursor is controlled by the movement of the pointer finger of either hand. Right-click is performed by touching the pointer finger and thumb together, and left-click is performed by touching the middle finger and thumb together.

zoom_test.py
Description: This script allows zooming in and out using hand gestures. Moving the thumbs on both hands farther apart zooms out, while moving the thumbs closer together zooms in.

Installation Instructions
To run these scripts, you'll need to install several Python packages. Use the following pip install commands to set up your environment:


pip install opencv-python

pip install mediapipe

pip install pyttsx3

pip install speechrecognition

pip install pyautogui

pip install numpy

pip install pyaudio

pip install pyobjc-framework-Quartz (MacOS only)

pip install pygetwindow (Windows only)
