from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import cv2
import mediapipe as mp
from collections import deque

app = Flask(__name__)
CORS(app)

# ==============================================
# MEDIAPIPE SETUP
# ==============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ==============================================
# MODEL LOADING
# ==============================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gesture_model.pkl')
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"[SignBridge] Model loaded! Classes: {model.classes_}")
    else:
        print("[SignBridge] No trained model found.")

# ==============================================
# PREDICTION SMOOTHING
# Keeps a rolling window of the last N predictions
# and returns the majority vote. Eliminates single-
# frame flickers (e.g. 4 flashing while showing 5).
# ==============================================
SMOOTH_WINDOW = 5
pred_buffer = deque(maxlen=SMOOTH_WINDOW)
conf_buffer = deque(maxlen=SMOOTH_WINDOW)

def smooth_prediction(gesture, confidence):
    pred_buffer.append(gesture)
    conf_buffer.append(confidence)

    counts = {}
    for g in pred_buffer:
        counts[g] = counts.get(g, 0) + 1

    majority = max(counts, key=counts.get)

    majority_confs = [
        conf_buffer[i]
        for i, g in enumerate(pred_buffer)
        if g == majority
    ]
    avg_conf = float(np.mean(majority_confs))
    return majority, avg_conf

# ==============================================
# PREDICT ENDPOINT
# Uses the exact same raw x,y,z landmark format
# your model was trained on — nothing changed there.
# Only addition is the smoothing buffer above.
# ==============================================
@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    file = request.files.get('frame')
    if not file:
        return jsonify({'gesture': 'NO_HAND', 'confidence': 0})

    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'gesture': 'NO_HAND', 'confidence': 0})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        pred_buffer.clear()
        conf_buffer.clear()
        return jsonify({'gesture': 'NO_HAND', 'confidence': 0})

    # Raw landmarks — same format your model was trained on
    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    if model is not None:
        features = np.array(landmarks).reshape(1, -1)
        raw_pred = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        raw_conf = float(max(probs))

        # Smooth over last 5 frames to kill flicker
        gesture, confidence = smooth_prediction(raw_pred, raw_conf)

        print(f"[SignBridge] Raw: {raw_pred} ({raw_conf:.1%})  Smoothed: {gesture} ({confidence:.1%})")
        return jsonify({'gesture': str(gesture), 'confidence': confidence})
    else:
        return jsonify({'gesture': 'Hand', 'confidence': 0.85})

# ==============================================
# HEALTH CHECK
# ==============================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': list(model.classes_) if model is not None else []
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)