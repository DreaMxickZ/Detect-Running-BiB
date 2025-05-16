from ultralytics import YOLO
import multiprocessing

print(f"CPU cores available: {multiprocessing.cpu_count()}")

# โหลดโมเดล
model = YOLO('yolov8n.pt')

# เริ่มเทรน
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,  # ปรับจาก 166 เป็น 16
    device='cpu',
    workers=6,
    name='bib_aug_yolo_default',
    augment=True  # ใช้ default augmentation ของ YOLOv8
)