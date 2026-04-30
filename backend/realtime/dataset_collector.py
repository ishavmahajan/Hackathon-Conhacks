import cv2
import mediapipe as mp
import csv
import os

print("Dataset Collector Running...")

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===== CSV Setup =====
os.makedirs("data/raw", exist_ok=True)
file_path = "data/raw/gestures.csv"

file = open(file_path, "a", newline="")
writer = csv.writer(file)

# ===== CAMERA (simple + reliable) =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    landmark_list = []

    # Draw + extract landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

    # Show window
    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # ===== SAVE DATA =====
    if key == ord('a') and landmark_list:
        writer.writerow(landmark_list + ["A"])
        print("Saved A")
        file.flush()

    if key == ord('b') and landmark_list:
        writer.writerow(landmark_list + ["B"])
        print("Saved B")
        file.flush()
    
    if key == ord('c') and landmark_list:
        writer.writerow(landmark_list + ["C"])
        print("Saved C")
        file.flush()

    if key == ord('d') and landmark_list:
        writer.writerow(landmark_list + ["D"])
        print("Saved D")
        file.flush()
    
    if key == ord('e') and landmark_list:
        writer.writerow(landmark_list + ["E"])
        print("Saved E")
        file.flush()

    if key == ord('f') and landmark_list:   
        writer.writerow(landmark_list + ["F"])
        print("Saved F")
        file.flush()

    if key == ord('g') and landmark_list:
        writer.writerow(landmark_list + ["G"])
        print("Saved G")
        file.flush()

    if key == ord('h') and landmark_list:
        writer.writerow(landmark_list + ["H"])
        print("Saved H")
        file.flush()

    if key == ord('i') and landmark_list:
        writer.writerow(landmark_list + ["I"])
        print("Saved I")
        file.flush()
        
    if key == ord('k') and landmark_list:
        writer.writerow(landmark_list + ["K"])
        print("Saved K")
        file.flush()
    
    if key == ord('l') and landmark_list:
        writer.writerow(landmark_list + ["L"])
        print("Saved L")
        file.flush()

    if key == ord('m') and landmark_list:
        writer.writerow(landmark_list + ["M"])
        print("Saved M")
        file.flush()

    if key == ord('n') and landmark_list:
        writer.writerow(landmark_list + ["N"])
        print("Saved N")
        file.flush()
    
    if key == ord('o') and landmark_list:
        writer.writerow(landmark_list + ["O"])
        print("Saved O")
        file.flush()
        
    if key == ord('p') and landmark_list:
        writer.writerow(landmark_list + ["P"])
        print("Saved P")
        file.flush()   
    
    if key == ord('q') and landmark_list:
        writer.writerow(landmark_list + ["Q"])
        print("Saved Q")
        file.flush()
    
    if key == ord('r') and landmark_list:
        writer.writerow(landmark_list + ["R"])
        print("Saved R")
        file.flush()

    if key == ord('s') and landmark_list:
        writer.writerow(landmark_list + ["S"])
        print("Saved S")
        file.flush()

    if key == ord('t') and landmark_list:
        writer.writerow(landmark_list + ["T"])
        print("Saved T")
        file.flush()

    if key == ord('u') and landmark_list:
        writer.writerow(landmark_list + ["U"])
        print("Saved U")
        file.flush()

    if key == ord('v') and landmark_list:
        writer.writerow(landmark_list + ["V"])
        print("Saved V")
        file.flush()    

    if key == ord('w') and landmark_list:
        writer.writerow(landmark_list + ["W"])
        print("Saved W")
        file.flush()

    if key == ord('x') and landmark_list:
        writer.writerow(landmark_list + ["X"])
        print("Saved X")
        file.flush()

    if key == ord('y') and landmark_list:
        writer.writerow(landmark_list + ["Y"])
        print("Saved Y")
        file.flush()

    # Quit
    if key == ord('z'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
