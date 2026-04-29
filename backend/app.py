from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import cv2
import mediapipe as mp

app = Flask(__name__)
CORS(app)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gesture_model.pkl')
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded!")
    else:
        print("No trained model found. Using placeholder.")

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    file = request.files['frame']
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'gesture': 'NO_HAND', 'confidence': 0})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return jsonify({'gesture': 'NO_HAND', 'confidence': 0})

    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    if model is not None:
        features = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))
        return jsonify({'gesture': prediction, 'confidence': confidence})
    else:
        return jsonify({'gesture': 'Hand', 'confidence': 0.85})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)