import cv2
import mediapipe as mp
import csv
import os

print("=== Number Collector (Both Hands Visible) ===")
print("Press 0-9 to save, Q to quit\n")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs("data/raw", exist_ok=True)
file_path = "data/raw/numbers_fixed.csv"

file = open(file_path, "a", newline="")
writer = csv.writer(file)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Only save first hand's landmarks
        for lm in results.multi_hand_landmarks[0].landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

    total = sum(1 for _ in open(file_path)) if os.path.exists(file_path) else 0
    cv2.putText(frame, f"Samples: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Number Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key >= ord('0') and key <= ord('9') and landmark_list:
        number = chr(key)
        writer.writerow(landmark_list + [number])
        print(f"Saved {number} (total: {total + 1})")
        file.flush()

    if key == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()

from collections import Counter
with open(file_path, 'r') as f:
    labels = [row[-1] for row in csv.reader(f)]
    counts = Counter(labels)
    print("\n=== Done ===")
    for label in sorted(counts.keys()):
        print(f"  {label}: {counts[label]}")