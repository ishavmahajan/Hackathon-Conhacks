import cv2
import mediapipe as mp
from model.predict import GesturePredictor
from collections import deque


class CameraDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0)
        self.predictor = GesturePredictor()
        self.predictions = deque(maxlen=10)

    def extract_normalized_landmarks(self, results):
        all_hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]

                hand_data = []

                for lm in hand_landmarks.landmark:
                    hand_data.append(lm.x - wrist.x)
                    hand_data.append(lm.y - wrist.y)
                    hand_data.append(lm.z - wrist.z)

                all_hands.append(hand_data)

        if len(all_hands) == 0:
            return None

        if len(all_hands) == 1:
            all_hands.append([0] * 63)

        if len(all_hands) > 2:
            all_hands = all_hands[:2]

        return all_hands[0] + all_hands[1]

    def run(self):
        print("Starting SignBridge prediction...")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb)

            detected_text = "No hand detected"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                landmarks = self.extract_normalized_landmarks(results)

                if landmarks is not None:
                    prediction, confidence = self.predictor.predict(landmarks)
                    detected_text = f"{prediction} ({confidence * 100:.1f}%)"
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