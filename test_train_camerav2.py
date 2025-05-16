import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime
import os
import time
import re
from collections import defaultdict

# ตั้งค่า
OUTPUT_DIR = "bib_logs2"
LOG_FILE = "bib_records2.csv"
CONF_THRESHOLD = 0.5
DUPLICATE_TIME_SEC = 10
VOTING_WINDOW_SEC = 3
VOTING_THRESHOLD = 2

BLACKLIST = ['m40', 'f30', 'fun', 'run', 'sponsor', 'nike', 'qr', 'km']
VALID_BIB_PATTERNS = [
    r'^5\d{3}$',
    r'^10\d{3}$',
    r'^21\d{3}$',
    r'^\d{4}$'
]

def is_valid_bib(bib):
    for pattern in VALID_BIB_PATTERNS:
        if re.match(pattern, bib):
            return True
    return False

def normalize_bib(text):
    text = text.strip().lstrip('0')
    return text if text.isdigit() else None

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO('runs/detect/bib_aug_yolo_default/weights/best.pt')
reader = easyocr.Reader(['en'])

last_detected = {}
detection_votes = defaultdict(list)  # bib_number: list[timestamp]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Start capturing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            conf = box.conf.item()
            if conf < 0.3:
                continue

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped_bib = frame[y1:y2, x1:x2]

            ocr_result = reader.readtext(cropped_bib)

            # ใช้แค่กล่องแรกเท่านั้น และต้องเป็นตัวเลขล้วน
            normalized = None
            if ocr_result:
                bbox, text, ocr_conf = ocr_result[0]
                if ocr_conf > CONF_THRESHOLD:
                    clean = text.strip().replace('O', '0').replace('I', '1')
                    if clean.isdigit():
                        normalized = clean.lstrip('0') or '0'

            if not normalized:
                continue

            current_time = time.time()

            if is_valid_bib(normalized):
                detection_votes[normalized].append(current_time)
                detection_votes[normalized] = [t for t in detection_votes[normalized] if current_time - t <= VOTING_WINDOW_SEC]

                if len(detection_votes[normalized]) >= VOTING_THRESHOLD:
                    last_time = last_detected.get(normalized, 0)
                    if current_time - last_time > DUPLICATE_TIME_SEC:
                        last_detected[normalized] = current_time

                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{normalized}_{timestamp_str}.jpg"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)

                        with open(LOG_FILE, "a") as f:
                            f.write(f"{timestamp_str},{normalized},{filename}\n")

                        print(f"Detected bib: {normalized}, saved image and log.")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, normalized, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Bib Detection Realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
