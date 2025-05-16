import os

# ใส่ path ที่ต้องการตรวจสอบ เช่นไฟล์หรือโฟลเดอร์
path_to_check = r"C:\Users\weera\OneDrive\เดสก์ท็อป\BIBScan\test_train.py"

# ตรวจสอบว่า path นี้มีอยู่จริงไหม
if os.path.exists(path_to_check):
    print(f"✅ Path นี้มีอยู่จริง: {path_to_check}")
else:
    print(f"❌ ไม่พบ path นี้: {path_to_check}")