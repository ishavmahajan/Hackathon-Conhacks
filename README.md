# SignBridge

SignBridge is a real-time sign language translator that uses a webcam to detect sign gestures and convert them into text. The project is designed to reduce communication barriers by making common signs easier to understand in everyday conversations.

## Overview

The system captures video from a webcam, extracts body and hand landmarks, converts those landmarks into feature vectors, and uses a trained machine learning model to predict the gesture in real time. The current prototype focuses on a small set of common signs such as **hello**, **please**, and **thanks**.

## Features

- Real-time webcam-based gesture recognition.
- Landmark extraction using MediaPipe Holistic / hand tracking pipeline.
- Live text output for detected signs.
- Optional text-to-speech support for spoken output.
- Lightweight machine learning pipeline using scikit-learn.

## How It Works

1. **Input:** The webcam captures live video frames.
2. **Landmark Detection:** MediaPipe detects pose and hand landmarks from each frame.
3. **Feature Extraction:** Landmark coordinates are normalized into a fixed-length feature vector.
4. **Prediction:** A trained `RandomForestClassifier` predicts the most likely sign from the extracted features.
5. **Output:** The predicted sign is displayed as text and can optionally be spoken aloud.

## Tech Stack

- **Frontend:** HTML
- **Backend:** Python
- **Computer Vision:** MediaPipe
- **Machine Learning:** scikit-learn RandomForestClassifier
- **Numerical Processing:** NumPy
- **Video Processing:** OpenCV
- **Optional Data Handling:** pandas

## Project Structure

```bash
SignBridge/
│
├── backend/
│   ├── app.py
│   ├── model/
│   │   ├── train_live.py
│   │   └── predict.py
│   ├── realtime/
│   │   └── camera.py
│   └── utils/
│       └── collect_live_data.py
│
├── data/
│   └── live_processed/
│
├── models/
│   └── gesture_model.pkl
│
└── README.md
```

A clean ML project structure usually separates source code, data, and saved models so the workflow is easier to maintain and debug.

## Installation

### Prerequisites

Make sure you have installed:
- Python 3.11+
- pip
- A working webcam

### Install dependencies

```bash
pip install numpy opencv-python mediapipe scikit-learn pandas joblib
```

> Note: MediaPipe version compatibility may vary depending on whether your code uses the legacy `solutions` API or the newer Tasks API.

## Usage

### 1. Collect gesture data

```bash
python backend/utils/collect_live_data.py
```

This script captures webcam data and saves processed gesture samples into `.npy` files for training.

### 2. Train the model

```bash
python backend/model/train_live.py
```

This trains the Random Forest classifier on the collected landmark data and saves the trained model to `models/gesture_model.pkl`.

### 3. Run the real-time translator

```bash
python backend/app.py
```

This starts the live webcam translator and displays predicted signs in real time.

## Current Limitations

- The model currently supports only a limited vocabulary.
- Performance depends on lighting, camera angle, and consistent signing position.
- Different MediaPipe versions may require code changes because of API differences between `solutions` and Tasks.
- The current prototype focuses on single-sign recognition rather than full sentence translation.

## Future Improvements

- Expand the sign vocabulary.
- Add sequence-based recognition for phrases and full sentences.
- Improve UI/UX for accessibility.
- Add stronger text-to-speech integration.
- Explore browser or video-call integration for real-world use.

## Why SignBridge

SignBridge is a simple but meaningful step toward more accessible communication. By combining computer vision and machine learning, the project shows how real-time sign recognition can become a practical assistive tool rather than just a research demo.

## License

This project is for educational and hackathon purposes. Update this section with your preferred open-source license if you plan to publish it.
