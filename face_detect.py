# mood_swing.py
# Author: Raiyan Rahman
# Date: March 02, 2020
# Description: Use the device's webcam to detect the faces present, in real
# time, and notify the mood of the faces.

import cv2
import sys

# Default cascade path.
DEFAULT_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Get the argument for the path to the cascade and create the classifier.
if len(sys.argv) == 2:
    cascade_path = sys.argv[1]
    face_detection_cascade = cv2.CascadeClassifier(cascade_path)
else:
    face_detection_cascade = cv2.CascadeClassifier(DEFAULT_CASCADE_PATH)

# Open the default webcam using openCV.
webcam = cv2.VideoCapture(0)

# Use an infinite loop to keep using the webcam.
while True:
    # Get the current frame from the webcam.
    ret, frame = webcam.read()

    # Search for faces in the captured frame.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Convert to grayscale

    detected_faces = face_detection_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.25,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around the detected faces in the frame.
    for (x, y, width, height) in detected_faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
    cv2.imshow('Feed', frame)

    # Allow exiting the program with the q-key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
