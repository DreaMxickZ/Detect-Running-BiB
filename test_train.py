from ultralytics import YOLO
import cv2
import os
import glob

# โหลดโมเดลที่ผ่านการฝึกแล้ว
model = YOLO("runs/detect/bib_aug_yolo_default/weights/best.pt")

# ทดสอบบนภาพใหม่
# โหลดภาพทั้งหมดในโฟลเดอร์ (รองรับ .jpg, .png)
image_paths = glob.glob("D:/pictureTest/*.jpg") + glob.glob("D:/pictureTest/*.png")

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
os.makedirs("predicted", exist_ok=True)

# วนลูปทำนายทุกภาพ
for path in image_paths:
    print(f"Predicting: {path}")
    results = model(path)

    # วาด bounding box ลงในภาพ
    annotated_img = results[0].plot()

    # สร้าง path สำหรับบันทึกผล
    out_path = f"predicted/{os.path.basename(path)}"
    cv2.imwrite(out_path, annotated_img)
    print(f"Saved: {out_path}")