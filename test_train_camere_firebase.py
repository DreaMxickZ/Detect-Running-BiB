import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import time
import os
import uuid
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import threading
import queue
from collections import defaultdict
import gc
import signal
import sys
import tempfile
import platform

# 🔧 กำหนดค่าหลัก
FIREBASE_CREDENTIAL = "firebase_key.json"
FIREBASE_BUCKET = "detech-bib-running.firebasestorage.app"
YOLO_MODEL_PATH = "runs/detect/bib_aug_yolo_default/weights/best.pt"

# ตัวแปรสำหรับควบคุมระบบ
detected_bibs = set()  # เก็บเลข bib ที่เจอแล้ว
bib_tracking = defaultdict(int)  # ติดตาม bib แต่ละตัวกี่เฟรม
upload_queue = queue.Queue(maxsize=50)  # จำกัดขนาดคิว
running = True
processing_lock = threading.Lock()  # ป้องกัน race condition

# 🎯 การตั้งค่าการตรวจจับ (ลดความซับซ้อน)
DETECTION_CONFIDENCE = 0.6  # เพิ่มขึ้นเล็กน้อย
OCR_CONFIDENCE = 0.7        # เพิ่มขึ้นเล็กน้อย
MIN_TRACKING_FRAMES = 2     # ลดลง
PROCESS_EVERY_N_FRAMES = 5  # เพิ่มขึ้น (ประมวลผลน้อยลง)
MAX_TRACKING_HISTORY = 100  # จำกัดจำนวน bib ที่ track
CAMERA_TIMEOUT = 5.0        # timeout สำหรับกล้อง

def signal_handler(sig, frame):
    """จัดการ signal เพื่อปิดโปรแกรมอย่างปลอดภัย"""
    global running
    print("\n🛑 Received interrupt signal, shutting down safely...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_temp_directory():
    """สร้างและตรวจสอบ temp directory ที่เหมาะสม"""
    try:
        # ใช้ tempfile เพื่อหา temp directory ที่เหมาะสม
        temp_dir = tempfile.gettempdir()
        
        # สร้าง subdirectory สำหรับแอปนี้
        app_temp_dir = os.path.join(temp_dir, "bib_detection")
        
        # สร้าง directory ถ้ายังไม่มี
        os.makedirs(app_temp_dir, exist_ok=True)
        
        # ทดสอบการเขียนไฟล์
        test_file = os.path.join(app_temp_dir, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Temp directory ready: {app_temp_dir}")
            return app_temp_dir
        except:
            # ถ้าเขียนไม่ได้ ใช้ current directory
            print(f"⚠️ Cannot write to {app_temp_dir}, using current directory")
            return os.path.join(os.getcwd(), "temp_images")
    except Exception as e:
        print(f"❌ Error setting up temp directory: {e}")
        # fallback ไปใช้ current directory
        fallback_dir = os.path.join(os.getcwd(), "temp_images")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir

def check_required_files():
    """ตรวจสอบไฟล์ที่จำเป็น"""
    missing_files = []
    
    if not os.path.exists(FIREBASE_CREDENTIAL):
        missing_files.append(FIREBASE_CREDENTIAL)
    
    if not os.path.exists(YOLO_MODEL_PATH):
        missing_files.append(YOLO_MODEL_PATH)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    return True

def init_firebase():
    """เริ่มต้นระบบ Firebase"""
    try:
        if not check_required_files():
            return None, None, None
        
        # ตรวจสอบว่า Firebase ได้ initialize แล้วหรือยัง
        try:
            firebase_admin.get_app()
            print("✅ Firebase already initialized")
        except ValueError:
            cred = credentials.Certificate(FIREBASE_CREDENTIAL)
            firebase_admin.initialize_app(cred, {
                'storageBucket': FIREBASE_BUCKET
            })
            print("✅ Firebase initialized successfully")
        
        db = firestore.client()
        bucket = storage.bucket()
        return db, bucket, True
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")
        return None, None, False

def load_models():
    """โหลดโมเดล YOLO และ OCR"""
    try:
        # ตั้งค่า YOLO ให้ใช้ทรัพยากรน้อยลง
        model = YOLO(YOLO_MODEL_PATH)
        model.overrides['verbose'] = False  # ลด log
        
        # ตั้งค่า OCR ให้ใช้ GPU ถ้ามี
        reader = easyocr.Reader(['en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
        print("✅ Models loaded successfully")
        return model, reader, True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return None, None, False

def clean_text(text):
    """ทำความสะอาดข้อความและตรวจสอบว่าเป็นหมายเลข bib หรือไม่"""
    if not text or len(text.strip()) == 0:
        return None
        
    # ลบตัวอักษรพิเศษ เหลือแค่ตัวเลขและตัวอักษร
    cleaned = ''.join(filter(str.isalnum, str(text).strip()))
    
    # ถ้าเป็นตัวเลขทั้งหมด
    if cleaned.isdigit() and len(cleaned) >= 1:
        return cleaned
    
    # ถ้ามีตัวอักษรปนอยู่ ให้เอาแค่ตัวเลข
    numbers_only = ''.join(filter(str.isdigit, cleaned))
    if numbers_only and len(numbers_only) >= 1:
        return numbers_only
    
    return None

def is_valid_bib_number(bib_number):
    """ตรวจสอบว่าเป็นหมายเลข bib ที่ถูกต้องหรือไม่"""
    if not bib_number or not str(bib_number).isdigit():
        return False
    
    # ตรวจสอบความยาว (ปกติจะเป็น 1-6 หลัก)
    if len(str(bib_number)) < 1 or len(str(bib_number)) > 6:
        return False
    
    # ตรวจสอบช่วงตัวเลข (ปรับตามงานของคุณ)
    try:
        bib_int = int(bib_number)
        if bib_int < 1 or bib_int > 99999:
            return False
    except ValueError:
        return False
    
    return True

def check_bib_exists(db, bib_number):
    """ตรวจสอบว่า bib มีอยู่ใน Firebase แล้วหรือไม่"""
    try:
        query = db.collection("runners").where(filter=firestore.FieldFilter("bib_number", "==", str(bib_number))).limit(1)
        docs = list(query.stream())
        return len(docs) > 0
    except Exception as e:
        print(f"❌ Error checking bib existence: {e}")
        return False

def save_image_safely(image, file_path, max_retries=3):
    """บันทึกภาพอย่างปลอดภัยพร้อม retry mechanism"""
    if image is None or image.size == 0:
        print("❌ Invalid image data")
        return False
    
    for attempt in range(max_retries):
        try:
            # ตรวจสอบว่า directory มีอยู่
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # ปรับปรุงคุณภาพภาพก่อนบันทึก
            if len(image.shape) == 3 and image.shape[2] == 3:  # BGR image
                # เพิ่มความคมชัด
                enhanced = cv2.convertScaleAbs(image, alpha=1.1, beta=5)
                
                # บันทึกด้วยคุณภาพสูง
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                success = cv2.imwrite(file_path, enhanced, encode_params)
            else:
                success = cv2.imwrite(file_path, image)
            
            if success and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return True
            else:
                print(f"⚠️ Image save attempt {attempt + 1} failed (file not created or empty)")
                
        except Exception as e:
            print(f"⚠️ Image save attempt {attempt + 1} error: {e}")
        
        # รอสักครู่ก่อน retry
        if attempt < max_retries - 1:
            time.sleep(0.1)
    
    print(f"❌ Failed to save image after {max_retries} attempts: {file_path}")
    return False

def upload_worker(db, bucket):
    """Worker thread สำหรับอัปโหลดข้อมูล - ปรับปรุงให้ทนทานขึ้น"""
    temp_dir = get_temp_directory()
    
    while running:
        try:
            # ใช้ timeout เพื่อไม่ให้ thread ค้าง
            upload_data = upload_queue.get(timeout=2.0)
            
            if upload_data is None:  # Signal to stop
                break
                
            bib_number = upload_data['bib_number']
            crop_image = upload_data['crop']
            confidence = upload_data['confidence']
            detection_time = upload_data['timestamp']
            
            # สร้างชื่อไฟล์
            unique_id = str(uuid.uuid4())[:8]
            filename = f"bib_{bib_number}_{int(detection_time)}_{unique_id}.jpg"
            local_path = os.path.join(temp_dir, filename)
            
            try:
                # บันทึกภาพชั่วคราวด้วยฟังก์ชันที่ปรับปรุงแล้ว
                success = save_image_safely(crop_image, local_path)
                if not success:
                    print(f"❌ Failed to save temp image for bib {bib_number}")
                    continue
                
                print(f"✅ Image saved locally: {filename} (Size: {os.path.getsize(local_path)} bytes)")
                
                # อัปโหลดไปยัง Firebase Storage
                blob = bucket.blob(f'bibs/{filename}')
                blob.upload_from_filename(local_path)
                blob.make_public()
                image_url = blob.public_url
                
                # บันทึกข้อมูลใน Firestore
                doc_data = {
                    "bib_number": str(bib_number),
                    "cp3time": datetime.fromtimestamp(detection_time).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "guntime": None,
                    "image_url": image_url,
                    "detection_confidence": float(confidence),
                    "processed_at": firestore.SERVER_TIMESTAMP,
                    "detection_timestamp": detection_time
                }
                
                db.collection("runners").add(doc_data)
                print(f"✅ Uploaded bib: {bib_number} (Confidence: {confidence:.2f})")
                
            except Exception as upload_error:
                print(f"❌ Upload error for bib {bib_number}: {upload_error}")
            finally:
                # ลบไฟล์ชั่วคราว
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        print(f"🗑️ Cleaned up temp file: {filename}")
                except Exception as cleanup_error:
                    print(f"⚠️ Failed to cleanup temp file {filename}: {cleanup_error}")
            
        except queue.Empty:
            continue
        except Exception as e:
            if running:  # แสดง error เฉพาะเมื่อยังทำงานอยู่
                print(f"❌ Upload worker error: {e}")
            continue
    
    print("🔄 Upload worker stopped")

def cleanup_tracking_data():
    """ทำความสะอาดข้อมูล tracking เก่า"""
    global bib_tracking
    
    if len(bib_tracking) > MAX_TRACKING_HISTORY:
        # เก็บแค่ bib ที่มี count สูงสุด
        sorted_bibs = sorted(bib_tracking.items(), key=lambda x: x[1], reverse=True)
        bib_tracking = defaultdict(int, dict(sorted_bibs[:MAX_TRACKING_HISTORY//2]))
        gc.collect()

def process_detections(frame, results, reader, db):
    """ประมวลผลการตรวจจับ - ปรับปรุงให้เร็วขึ้น"""
    if not running:
        return
        
    try:
        with processing_lock:
            current_frame_bibs = set()
            
            # จำกัดจำนวนการตรวจจับที่ประมวลผล
            boxes = results.boxes.data[:5] if len(results.boxes.data) > 5 else results.boxes.data
            
            for i, box in enumerate(boxes):
                if not running:
                    break
                    
                x1, y1, x2, y2, score, class_id = box.tolist()
                
                if score < DETECTION_CONFIDENCE:
                    continue
                
                # วาดกรอบ
                color = (0, 255, 0) if score > 0.7 else (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"Det {i+1}: {score:.2f}", 
                           (int(x1), int(y1-30)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # ตัดภาพ bib - ปรับปรุงการตัดและตรวจสอบขนาด
                padding = 5  # ลด padding
                x1_pad = max(0, int(x1) - padding)
                y1_pad = max(0, int(y1) - padding)
                x2_pad = min(frame.shape[1], int(x2) + padding)
                y2_pad = min(frame.shape[0], int(y2) + padding)
                
                # ตรวจสอบขนาดก่อนตัด
                crop_width = x2_pad - x1_pad
                crop_height = y2_pad - y1_pad
                
                if crop_width < 20 or crop_height < 20:
                    print(f"⚠️ Crop too small: {crop_width}x{crop_height}")
                    continue
                
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop is None or crop.size == 0:
                    print("⚠️ Empty crop image")
                    continue
                
                # OCR - ลดความซับซ้อน
                try:
                    # ปรับปรุงภาพแบบเร็ว
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    
                    # เพิ่มความคมชัดแบบเร็ว
                    enhanced = cv2.convertScaleAbs(gray_crop, alpha=1.2, beta=10)
                    
                    # OCR
                    ocr_results = reader.readtext(enhanced, paragraph=False)
                    
                    best_bib = None
                    best_confidence = 0
                    
                    for (bbox, text, conf) in ocr_results:
                        if conf > OCR_CONFIDENCE:
                            cleaned_text = clean_text(text)
                            
                            if cleaned_text and is_valid_bib_number(cleaned_text):
                                if conf > best_confidence:
                                    best_bib = cleaned_text
                                    best_confidence = conf
                    
                    if best_bib:
                        current_frame_bibs.add(best_bib)
                        bib_tracking[best_bib] += 1
                        
                        # แสดงข้อความ
                        text_color = (0, 255, 0) if best_bib in detected_bibs else (0, 0, 255)
                        cv2.putText(frame, f"BIB: {best_bib} ({bib_tracking[best_bib]})", 
                                   (int(x1), int(y1-10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                        
                        # ตรวจสอบเงื่อนไขการบันทึก
                        if (bib_tracking[best_bib] >= MIN_TRACKING_FRAMES and 
                            best_bib not in detected_bibs):
                            
                            # ตรวจสอบใน Firebase
                            if not check_bib_exists(db, best_bib):
                                detected_bibs.add(best_bib)
                                
                                # เพิ่มเข้าคิวอัปโหลด (non-blocking)
                                upload_data = {
                                    'bib_number': best_bib,
                                    'crop': frame.copy(),
                                    'confidence': score,
                                    'timestamp': time.time()
                                }
                                
                                try:
                                    upload_queue.put_nowait(upload_data)
                                    print(f"🎯 NEW BIB: {best_bib} (YOLO: {score:.2f}, OCR: {best_confidence:.2f})")
                                except queue.Full:
                                    print("⚠️ Upload queue full, skipping...")
                            else:
                                detected_bibs.add(best_bib)
                                print(f"⚠️ Bib {best_bib} already exists")
                                
                except Exception as ocr_error:
                    cv2.putText(frame, "OCR Failed", 
                               (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    print(f"❌ OCR error: {ocr_error}")
                    continue
            
            # ลดค่า tracking
            bibs_to_reduce = set(bib_tracking.keys()) - current_frame_bibs
            for bib in list(bibs_to_reduce):  # ใช้ list() เพื่อป้องกัน dictionary size change
                bib_tracking[bib] = max(0, bib_tracking[bib] - 1)
                if bib_tracking[bib] == 0:
                    del bib_tracking[bib]
            
            # ทำความสะอาดข้อมูลเก่า
            if len(bib_tracking) > MAX_TRACKING_HISTORY:
                cleanup_tracking_data()
                
    except Exception as e:
        if running:
            print(f"❌ Processing error: {e}")

def setup_camera():
    """ตั้งค่ากล้องอย่างระมัดระวัง - แก้ปัญหา MSMF"""
    cap = None
    
    # ลองหลาย backend และหลายกล้อง
    backends = [
        cv2.CAP_DSHOW,    # DirectShow (Windows)
        cv2.CAP_MSMF,     # Microsoft Media Foundation
        cv2.CAP_V4L2,     # Video4Linux (Linux)
        cv2.CAP_ANY       # Auto detect
    ]
    
    camera_indices = [0, 1, 2]  # ลองกล้อง index 0, 1, 2
    
    for backend in backends:
        for camera_idx in camera_indices:
            try:
                print(f"🔍 Trying camera {camera_idx} with backend {backend}")
                cap = cv2.VideoCapture(camera_idx, backend)
                
                if not cap.isOpened():
                    if cap:
                        cap.release()
                    continue
                
                # ตั้งค่ากล้องทีละขั้น
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # ให้เวลากล้องเตรียมตัว
                    time.sleep(1)
                    
                    # ทดสอบอ่านเฟรม 3 ครั้ง
                    success_count = 0
                    for i in range(3):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            success_count += 1
                        time.sleep(0.1)
                    
                    if success_count >= 2:  # สำเร็จอย่างน้อย 2/3 ครั้ง
                        print(f"✅ Camera {camera_idx} ready (backend: {backend})")
                        print(f"📐 Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                        print(f"📊 FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
                        return cap
                    else:
                        print(f"❌ Camera {camera_idx} can't read frames consistently")
                        cap.release()
                        cap = None
                        
                except Exception as setup_error:
                    print(f"❌ Camera {camera_idx} setup error: {setup_error}")
                    if cap:
                        cap.release()
                    cap = None
                    continue
                    
            except Exception as e:
                print(f"❌ Camera {camera_idx} backend {backend} error: {e}")
                if cap:
                    cap.release()
                cap = None
                continue
    
    print("❌ No working camera found!")
    print("💡 Try these solutions:")
    print("   1. Check if camera is being used by another app")
    print("   2. Try different USB port")
    print("   3. Update camera drivers")
    print("   4. Run as administrator")
    print("   5. Check Windows Camera privacy settings")
    
    return None

def main():
    global running
    
    print("🚀 Starting Enhanced Real-time BIB Detection System...")
    print(f"🎯 Detection Settings:")
    print(f"   - YOLO Confidence: {DETECTION_CONFIDENCE}")
    print(f"   - OCR Confidence: {OCR_CONFIDENCE}")
    print(f"   - Min Tracking Frames: {MIN_TRACKING_FRAMES}")
    print(f"   - Process Every N Frames: {PROCESS_EVERY_N_FRAMES}")
    print(f"🖥️ Platform: {platform.system()} {platform.release()}")
    
    # เริ่มต้นระบบ
    db, bucket, firebase_ok = init_firebase()
    model, reader, models_ok = load_models()
    
    if not firebase_ok or not models_ok:
        print("❌ System initialization failed!")
        return
    
    # เริ่มต้น upload worker thread
    upload_thread = threading.Thread(target=upload_worker, args=(db, bucket))
    upload_thread.daemon = True
    upload_thread.start()
    
    # ตั้งค่ากล้อง
    cap = setup_camera()
    if cap is None:
        running = False
        return
    
    print("🎯 Starting BIB detection...")
    print("📝 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 'r' to reset detected bibs")
    print("   - Press 'c' to clear tracking")
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    last_cleanup = time.time()
    
    try:
        while running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("❌ Failed to read frame, retrying...")
                time.sleep(0.1)
                
                # ลองอ่านใหม่ 3 ครั้ง
                retry_count = 0
                while retry_count < 3 and running:
                    time.sleep(0.2)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                    retry_count += 1
                    print(f"🔄 Retry reading frame {retry_count}/3")
                
                if not ret or frame is None:
                    print("❌ Camera connection lost, trying to reconnect...")
                    cap.release()
                    time.sleep(2)
                    
                    # ลองเชื่อมต่อกล้องใหม่
                    cap = setup_camera()
                    if cap is None:
                        print("❌ Cannot reconnect camera, stopping...")
                        break
                    continue
            
            frame_count += 1
            current_time = time.time()
            
            # ประมวลผลทุก N เฟรม
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                try:
                    results = model(frame, verbose=False)[0]
                    process_detections(frame, results, reader, db)
                except Exception as e:
                    print(f"❌ Detection error: {e}")
            
            # คำนวณ FPS
            if current_time - fps_time >= 1.0:
                fps = frame_count / (current_time - fps_time)
                frame_count = 0
                fps_time = current_time
            
            # แสดงข้อมูลสถิติ
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detected: {len(detected_bibs)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracking: {len(bib_tracking)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Queue: {upload_queue.qsize()}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # แสดงผล
            try:
                cv2.imshow('BIB Detection System', frame)
            except cv2.error as cv_error:
                print(f"❌ Display error: {cv_error}")
                continue
            
            # จัดการ keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                with processing_lock:
                    detected_bibs.clear()
                print("🔄 Reset detected bibs")
            elif key == ord('c'):
                with processing_lock:
                    bib_tracking.clear()
                print("🔄 Clear tracking data")
            
            # ทำความสะอาดระยะๆ
            if current_time - last_cleanup > 30:  # ทุก 30 วินาที
                cleanup_tracking_data()
                gc.collect()
                last_cleanup = current_time
                
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Main loop error: {e}")
    finally:
        # ปิดระบบอย่างปลอดภัย
        print("🔄 Shutting down system...")
        running = False
        
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        # ส่งสัญญาณให้ upload worker หยุด
        try:
            upload_queue.put_nowait(None)
        except:
            pass
        
        # รอให้ upload thread จบ
        if upload_thread.is_alive():
            upload_thread.join(timeout=5)
        
        print("🎉 System stopped safely")
        print(f"📊 Total detected bibs: {len(detected_bibs)}")
        if detected_bibs:
            sorted_bibs = sorted([int(x) for x in detected_bibs if x.isdigit()])
            print(f"📝 Detected numbers: {sorted_bibs}")

if __name__ == '__main__':
    main()