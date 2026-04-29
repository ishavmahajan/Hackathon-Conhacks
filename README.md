# Sign Language Translator
## Overview

This project is a real time Sign Language Translator that detects hand gestures from a webcam, converts them into text, and optionally reads the text aloud using speech synthesis.

The goal is to reduce communication barriers by providing an accessible tool for translating sign language into spoken and written communication.

# Features
- Real-time hand tracking using a webcam
- Gesture recognition (limited set of signs)
- Live text output
- Optional text to speech

# How It Works
- Input: Webcam captures hand movements
- Processing: Hand landmarks are extracted using MediaPipe Hands
- Model: A trained classifier predicts the gesture based on landmark data
- Output: Displays translated text and optionally converts text to speech

# Tech Stack
- Frontend: HTML
- Backend: Python
- Computer Vision: MediaPipe Hands
- (Update this as we go)
