import cv2
import mediapipe as mp
import os
import sys

# Add the 'Backend' folder to sys.path so we can import 'model'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import GesturePredictor
from collections import deque


class CameraDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_draw = mp.solutions.drawing_utils

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # Reduced from 1 to 0 for much faster processing
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0)
        self.predictor = GesturePredictor()
        self.predictions = deque(maxlen=10)

    def extract_hand_data(self, hand_landmarks, anchor):
        if hand_landmarks is None:
            return [0.0] * 63

        hand_data = []
        for lm in hand_landmarks.landmark:
            hand_data.append(lm.x - anchor.x)
            hand_data.append(lm.y - anchor.y)
            hand_data.append(lm.z - anchor.z)
        return hand_data

    def extract_pose_data(self, pose_landmarks):
        selected_indices = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists

        if pose_landmarks is None:
            return [0.0] * (len(selected_indices) * 3)

        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]

        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        center_z = (left_shoulder.z + right_shoulder.z) / 2

        pose_data = []
        for idx in selected_indices:
            lm = pose_landmarks.landmark[idx]
            pose_data.append(lm.x - center_x)
            pose_data.append(lm.y - center_y)
            pose_data.append(lm.z - center_z)

        return pose_data

    def extract_normalized_landmarks(self, results):
        pose_data = self.extract_pose_data(results.pose_landmarks)

        left_anchor = None
        right_anchor = None

        if results.pose_landmarks:
            left_anchor = results.pose_landmarks.landmark[11]
            right_anchor = results.pose_landmarks.landmark[12]

        left_hand_data = self.extract_hand_data(results.left_hand_landmarks, left_anchor) if left_anchor else [0.0] * 63
        right_hand_data = self.extract_hand_data(results.right_hand_landmarks, right_anchor) if right_anchor else [0.0] * 63

        if (
            results.pose_landmarks is None
            and results.left_hand_landmarks is None
            and results.right_hand_landmarks is None
        ):
            return None

        features = pose_data + left_hand_data + right_hand_data

        # Force fixed length: 147 = 21 (pose) + 63 (left hand) + 63 (right hand)
        expected_len = 147
        if len(features) < expected_len:
            features += [0.0] * (expected_len - len(features))
        elif len(features) > expected_len:
            features = features[:expected_len]

        return features

    def run(self):
        print("Starting SignBridge prediction with Holistic...")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)

            detected_text = "No body/hands detected"

            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS
                )

            if results.left_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS
                )

            if results.right_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS
                )

            landmarks = self.extract_normalized_landmarks(results)

            if landmarks is not None:
                try:
                    prediction, confidence = self.predictor.predict(landmarks)
                    self.predictions.append(prediction)
                    final_prediction = max(set(self.predictions), key=self.predictions.count)
                    detected_text = f"{final_prediction} ({confidence * 100:.1f}%)"
                except Exception as e:
                    detected_text = "Prediction error"
                    print("Prediction error:", e)

            cv2.putText(
                frame,
                f"Detected: {detected_text}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("SignBridge - Real Time ASL", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if cv2.getWindowProperty("SignBridge - Real Time ASL", cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()

class FrameProcessor(CameraDetector):
    def __init__(self):
        super().__init__()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        landmarks = self.extract_normalized_landmarks(results)
        
        if landmarks is not None:
            try:
                prediction, confidence = self.predictor.predict(landmarks)
                self.predictions.append(prediction)
                final_prediction = max(set(self.predictions), key=self.predictions.count)
                return final_prediction, confidence
            except Exception as e:
                print("Prediction error:", e)
                return None, None
        return None, None

if __name__ == "__main__":
    detector = CameraDetector()
    detector.run()
