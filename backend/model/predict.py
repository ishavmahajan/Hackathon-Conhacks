import joblib
import numpy as np

class GesturePredictor:
    def __init__(self, model_path="models/gesture_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, landmarks):
        data = np.array(landmarks).reshape(1, -1)
        prediction = self.model.predict(data)[0]

        confidence = max(self.model.predict_proba(data)[0])
        return prediction, confidence