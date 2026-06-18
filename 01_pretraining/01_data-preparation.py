"""
สคริปต์นี้แบ่งข้อมูลดิบเป็น Train/Valid สำหรับ Pretraining

📌 ค่า config ที่สำคัญ:
  • split_idx = int(len(paragraphs) * 0.7) → แบ่ง 70% Train, 30% Valid
    - ถ้า Train มาก: โมเดลเรียนรู้ข้อมูลได้เยอะ แต่ Validate น้อย
    - ถ้า Valid มาก: ประเมินผลแม่นยำ แต่ข้อมูลสอนน้อย
    - 70/30 ใช้เมื่อข้อมูลน้อย เพื่อให้ Validation มีอย่างน้อย 2 ตัวอย่าง

💡 ผลต่อ Inference:
  - ข้อมูล Train คือสิ่งที่โมเดล "จำ" ได้
  - ถ้าข้อมูลมีคุณภาพสูง → คำตอบแม่นยำ
  - ถ้าข้อมูลมีเสียงรบกวน → คำตอบผิดเพี้ยน
  - การแบ่งข้อมูลที่ดีช่วยให้โมเดลไม่ “จำเกิน” (overfit)
"""
import json
import os

# 1. อ่านไฟล์ข้อความดิบ
with open("./dataset/ex-data-quality.txt", "r", encoding="utf-8") as f:
    paragraphs = [line.strip() for line in f if line.strip()]

# 2. แบ่งข้อมูลเป็น Train (70%) และ Valid (30%)
# เปลี่ยนจาก 80/20 เป็น 70/30 เพื่อให้ Validation มีอย่างน้อย 2 ตัวอย่าง
split_idx = int(len(paragraphs) * 0.7)
train_data = paragraphs[:split_idx]
valid_data = paragraphs[split_idx:]

print(f"📊 ข้อมูลทั้งหมด: {len(paragraphs)} ย่อหน้า")
print(f"📊 Train: {len(train_data)} ย่อหน้า")
print(f"📊 Valid: {len(valid_data)} ย่อหน้า")

# 3. ฟังก์ชันสำหรับบันทึกเป็น JSONL
def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for text in data:
            # สร้าง Dictionary ให้อยู่ในฟอร์แมต {"text": "..."}
            json_line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(json_line + "\n")

# บันทึกไฟล์ลงในโฟลเดอร์ data/
os.makedirs("data", exist_ok=True)
save_jsonl(train_data, "./data/raw/train.jsonl")
save_jsonl(valid_data, "./data/raw/valid.jsonl")