import cv2
from ultralytics import YOLO
import time


model = YOLO("model/letters.pt")   

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera error")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

 
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    results = model(frame, conf=0.5, verbose=False)

   
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Simple Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
