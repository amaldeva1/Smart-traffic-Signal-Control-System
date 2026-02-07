import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load traffic video
cap = cv2.VideoCapture("videos/traffic.mp4")

if not cap.isOpened():
    print("Video not opening")
    exit()

def get_density(count):
    if count <= 5:
        return "LOW"
    elif count <= 15:
        return "MEDIUM"
    else:
        return "HIGH"

# ---- DYNAMIC SIGNAL UPDATE ----
update_interval = 5  # seconds
last_update_time = time.time()
active_road = "ROAD 1"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Vehicle count per road
    roads = {
        "ROAD 1": 0,
        "ROAD 2": 0,
        "ROAD 3": 0
    }

    results = model(frame, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in [2, 3, 5, 7] and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2

                if cx < w // 3:
                    road = "ROAD 1"
                elif cx < 2 * w // 3:
                    road = "ROAD 2"
                else:
                    road = "ROAD 3"

                roads[road] += 1

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # ---- UPDATE ACTIVE ROAD BASED ON MAX COUNT (NOT SYSTEMATIC) ----
    current_time = time.time()
    if current_time - last_update_time >= update_interval:
        active_road = max(roads, key=roads.get)
        last_update_time = current_time

    # ---- COUNTDOWN TIMER ----
    time_left = int(update_interval - (current_time - last_update_time))
    if time_left < 0:
        time_left = 0

    cv2.putText(
        frame,
        f"NEXT SWITCH IN: {time_left}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Display road-wise info
    y = 60
    for road in ["ROAD 1", "ROAD 2", "ROAD 3"]:
        count = roads[road]
        density = get_density(count)
        signal = "GREEN" if road == active_road else "RED"

        cv2.putText(
            frame,
            f"{road} | Vehicles: {count} | Density: {density} | Signal: {signal}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if road == active_road else (0, 0, 255),
            2
        )
        y += 35

    # ACTIVE ROAD display
    cv2.putText(
        frame,
        f"ACTIVE ROAD: {active_road}",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Smart Traffic Signal Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
