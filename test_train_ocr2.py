from ultralytics import YOLO
import cv2
import os
import glob
import easyocr
import re

# โหลดโมเดล YOLO
model = YOLO("runs/detect/bib_aug_yolo_default/weights/best.pt")

# สร้าง OCR reader
reader = easyocr.Reader(['en'])

# โหลดภาพ
image_paths = glob.glob("D:/Running/*.jpg") + glob.glob("D:/Running/*.png")

# สร้างโฟลเดอร์เก็บผลลัพธ์
os.makedirs("predicted", exist_ok=True)

# วนลูปทำนาย
for path in image_paths:
    print(f"🔍 Predicting: {path}")
    results = model(path)

    # โหลดภาพต้นฉบับ
    img = cv2.imread(path)

    # วนลูปผลการตรวจจับทั้งหมด
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัด bbox
        cropped = img[y1:y2, x1:x2]  # ตัด bib ออกมา

        # OCR
        ocr_result = reader.readtext(cropped)
        bib_number = None

        for detection in ocr_result:
            text = detection[1].replace(" ", "")  # ลบช่องว่าง
            conf = detection[2]

            # กรองเฉพาะตัวเลขล้วน
            if re.fullmatch(r'\d+', text):
                bib_number = text
                print(f"✅ BIB Detected: {text} (Confidence: {conf:.2f})")

                # ใส่ข้อความบนภาพ
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3, cv2.LINE_AA)
                break
            else:
                print(f"❌ Skipped non-numeric text: {text}")

        # ถ้ายังไม่มีเลข ก็วาดกรอบเปล่า
        if not bib_number:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # บันทึกภาพที่มีกรอบและเลข
    out_path = f"predicted/{os.path.basename(path)}"
    cv2.imwrite(out_path, img)
    print(f"💾 Saved: {out_path}\n")
