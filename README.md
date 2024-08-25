![Swypeimage](https://github.com/user-attachments/assets/71f617cb-92d1-4f8e-99bc-d87a841dbb8c)

https://github.com/user-attachments/assets/48d37b46-a3da-4bed-9b96-32418d217977

## What is Swype AI?

Swype AI is an innovative assistive technology system designed to empower individuals with motor disabilities by allowing them to control their computers through natural hand gestures and voice commands. In less than 3 months, we transformed Swype from a simple idea into a fully functional product.

### Key Achievements

- **Board of Advisors:** Formed a board of advisors and are mentored by professors and university students from Harvard University, John Hopkins University, and the University of Maryland.
- **International Recognition:** Invited to an all-expenses-paid trip to present our product at the International Annual Patient Conference in Baltimore, Maryland, and featured on their app.
- **Publication Feature:** Will be featured in the American Chronic Pain Association’s Chronicle publication for the September Edition.
- **Collaborations:** Spoke with the Center for Accessible Technology and the Arthritis Foundation about our product.
- **Partnership:** Partnered with the Virginia Accessible Technology System to demo our product with real users.
- **Pitch Competition:** Invited to San Francisco for a pitch competition from Buildspace (7.4% graduation out of 70,000 students).
- **Beta Testing:** Opened up beta testing through our landing page to real users.

### Links

- **Landing Page:** [Swype Landing Page](https://swype.my.canva.site/)
- **Pitch Deck:** [Swype Pitch Deck](https://drive.google.com/file/d/15wBvCpavXI4JgsW5rkl6LBcOFVhh7WRv/view?usp=sharing)

## Files and Descriptions

### `gesture_control_test.py`
**Description:** This script enables cursor control using hand gestures. The cursor is controlled by the movement of the pointer finger of either hand. Right-click is performed by touching the pointer finger and thumb together, and left-click is performed by touching the middle finger and thumb together.

### `voice_control_test.py`
**Description:** This script acts as an AI voice assistant, allowing you to control various computer functions using voice commands. It can execute tasks such as navigating between tabs, controlling media playback, adjusting volume, copying and pasting text, and more. The assistant provides real-time feedback after executing each command and can be easily activated by speaking predefined commands.

### `fist_audio_test.py`
**Description:** This script allows you to activate audio by making a fist with your hand for 3 seconds. Once activated, it continues to analyze speech and convert it to text until no audio is registered for 10 seconds. Additionally, saying "Search" will automatically navigate to the search bar.

### `zoom_test.py`
**Description:** This script allows zooming in and out using hand gestures. Moving the thumbs on both hands farther apart zooms out, while moving the thumbs closer together zooms in.

## Installation Instructions

To run these scripts, you'll need to install several Python packages. Use the following pip install commands to set up your environment:

```bash
pip install opencv-python
pip install mediapipe
pip install pyttsx3
pip install speechrecognition
pip install pyautogui
pip install numpy
pip install pyaudio
pip install pyobjc-framework-Quartz  # MacOS only
pip install pygetwindow  # Windows only
