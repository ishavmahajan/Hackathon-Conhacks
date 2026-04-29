import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

EXCEL_PATH = "data/wlasl/ASL_combined_dataset.xlsx"
VIDEOS_DIR = "data/wlasl/videos"
OUTPUT_DIR = "data/processed"

target_words = ["hello", "yes", "no", "please", "sorry", "help", "good"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(EXCEL_PATH, sheet_name="WLASL")
df["gloss"] = df["gloss"].astype(str).str.lower()
df = df[df["gloss"].isin(target_words)]

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_normalized_landmarks(results):
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

    # Each hand = 21 landmarks * 3 = 63 values
    # Two hands = 126 values
    if len(all_hands) == 0:
        return None

    if len(all_hands) == 1:
        all_hands.append([0] * 63)

    if len(all_hands) > 2:
        all_hands = all_hands[:2]

    return all_hands[0] + all_hands[1]

data = []
labels = []

print("Starting normalized 2-hand extraction...")

for _, row in df.iterrows():
    video_id = str(row["video_id"]).zfill(5)
    label = row["gloss"]
    video_path = os.path.join(VIDEOS_DIR, video_id + ".mp4")

    if not os.path.exists(video_path):
        continue

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = extract_normalized_landmarks(results)

        if landmarks is not None:
            data.append(landmarks)
            labels.append(label)

        frame_count += 1

        if frame_count > 40:
            break

    cap.release()

X = np.array(data)
y = np.array(labels)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

print("Done!")
print("X shape:", X.shape)
print("y shape:", y.shape)