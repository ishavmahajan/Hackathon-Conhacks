import cv2
import mediapipe as mp
import numpy as np
import os


SAVE_DIR = "data/live_processed"
os.makedirs(SAVE_DIR, exist_ok=True)

label = input("Enter sign label: ").lower().strip()

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
data = []


def extract_hand_data(hand_landmarks, anchor):
    if hand_landmarks is None or anchor is None:
        return [0.0] * 63

    hand_data = []
    for lm in hand_landmarks.landmark:
        hand_data.extend([
            lm.x - anchor.x,
            lm.y - anchor.y,
            lm.z - anchor.z
        ])
    return hand_data


def extract_pose_data(pose_landmarks):
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
        pose_data.extend([
            lm.x - center_x,
            lm.y - center_y,
            lm.z - center_z
        ])

    return pose_data


def extract_landmarks(results):
    if (
        results.pose_landmarks is None
        and results.left_hand_landmarks is None
        and results.right_hand_landmarks is None
    ):
        return None

    pose_data = extract_pose_data(results.pose_landmarks)

    left_anchor = None
    right_anchor = None

    if results.pose_landmarks:
        left_anchor = results.pose_landmarks.landmark[11]
        right_anchor = results.pose_landmarks.landmark[12]

    left_hand_data = extract_hand_data(results.left_hand_landmarks, left_anchor)
    right_hand_data = extract_hand_data(results.right_hand_landmarks, right_anchor)

    return pose_data + left_hand_data + right_hand_data


print("Press S to save frames. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    cv2.putText(
        frame,
        f"Label: {label} | Samples: {len(data)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Live Data - Holistic", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        landmarks = extract_landmarks(results)
        if landmarks is not None:
            data.append(landmarks)
            print("Saved sample:", len(data), "Feature length:", len(landmarks))
        else:
            print("No pose/hands detected")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()

X_path = os.path.join(SAVE_DIR, f"{label}_X.npy")
np.save(X_path, np.array(data, dtype=np.float32))

print(f"Saved {len(data)} samples for {label} to {X_path}")
