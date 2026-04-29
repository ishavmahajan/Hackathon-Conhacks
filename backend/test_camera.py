import cv2

print("Starting camera test...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press Q to quit
    if key == ord('q'):
        break

    # If user closes window manually (X button)
    if cv2.getWindowProperty("Camera Test", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Camera closed properly")