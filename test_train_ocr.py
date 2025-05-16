from ultralytics import YOLO
import cv2
import os
import glob
import easyocr

# โหลดโมเดล YOLO
model = YOLO("runs/detect/bib_aug_yolo_default/weights/best.pt")

# สร้าง OCR reader
reader = easyocr.Reader(['en'])

# โหลดภาพ
image_paths = glob.glob("D:/pictureTest/*.jpg") + glob.glob("D:/pictureTest/*.png")

# สร้างโฟลเดอร์เก็บผลลัพธ์
os.makedirs("predicted", exist_ok=True)

# วนลูปทำนาย
for path in image_paths:
    print(f"Predicting: {path}")
    results = model(path)

    # โหลดภาพต้นฉบับ
    img = cv2.imread(path)

    # วนลูปผลการตรวจจับทั้งหมด
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัด bbox
        cropped = img[y1:y2, x1:x2]  # ตัดส่วนของ bib ออกมา

        # OCR
        ocr_result = reader.readtext(cropped)
        for detection in ocr_result:
            text = detection[1]  # ได้ข้อความที่ตรวจเจอ
            conf = detection[2]
            print(f"Detected Text: {text} (Confidence: {conf:.2f})")

            # วาดผลบนภาพ
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # วาดกรอบ
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # บันทึกภาพ
    out_path = f"predicted/{os.path.basename(path)}"
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
