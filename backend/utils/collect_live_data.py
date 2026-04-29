import cv2
import mediapipe as mp
import numpy as np
import os

SAVE_DIR = "data/live_processed"
os.makedirs(SAVE_DIR, exist_ok=True)

label = input("Enter sign label: ").lower()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

data = []

def extract_landmarks(results):
    all_hands = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            hand_data = []

            for lm in hand_landmarks.landmark:
                hand_data.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            all_hands.append(hand_data)

    if len(all_hands) == 0:
        return None

    if len(all_hands) == 1:
        all_hands.append([0] * 63)

    return all_hands[0] + all_hands[1]

print("Press S to save frames. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Label: {label} | Samples: {len(data)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collect Live Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        landmarks = extract_landmarks(results)
        if landmarks is not None:
            data.append(landmarks)
            print("Saved sample:", len(data))

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

X_path = os.path.join(SAVE_DIR, f"{label}_X.npy")

np.save(X_path, np.array(data))

print(f"Saved {len(data)} samples for {label}")