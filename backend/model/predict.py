import joblib
import numpy as np


class GesturePredictor:
    def __init__(self, model_path="models/gesture_model.pkl"):
        self.model = joblib.load(model_path)
        print("Model expects features:", self.model.n_features_in_)

    def predict(self, landmarks):
        data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        if data.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Feature length mismatch: got {data.shape[1]}, expected {self.model.n_features_in_}"
            )
        # Single call — get both label and confidence from predict_proba
        proba = self.model.predict_proba(data)[0]
        best_idx = int(np.argmax(proba))
        prediction = self.model.classes_[best_idx]
        confidence = float(proba[best_idx])
        return prediction, confidence
