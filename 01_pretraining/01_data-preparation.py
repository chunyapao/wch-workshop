"""
สคริปต์นี้เตรียมข้อมูลดิบสำหรับ Continued Pre-training

📌 ขั้นตอนการทำงาน:
  1. อ่านข้อความดิบจาก ex-data-quality.txt
  2. กรองบรรทัดที่สั้นเกินไป (< 40 ตัวอักษร) เช่น ชื่อ, อีเมล, URL
     เพราะไม่ใช่ประโยคที่มีเนื้อหา → โมเดลจะเรียนรู้ข้อความไร้สาระ
  3. รวมย่อหน้าสั้นๆ เข้าด้วยกัน ให้ได้ย่อหน้าที่สมบูรณ์ (≥ 120 ตัวอักษร)
     เพื่อให้แต่ละ block ตอนเทรนมีบริบทยาว → โมเดลตอบยาวขึ้น
  4. แบ่งเป็น Train (90%) / Valid (10%)

💡 ทำไมต้องรวมย่อหน้า?
  - ข้อมูลดิบแต่ละบรรทัดสั้นมาก (20-80 ตัวอักษร)
  - ถ้าไม่รวม → แต่ละ block ตอนเทรนมีแต่ข้อความสั้น + eos ถี่ → โมเดลตอบสั้น
  - รวมแล้ว → แต่ละ block มีเนื้อหาสมบูรณ์ → โมเดลเรียนรู้ประโยคยาว → ตอบยาว
"""
import json
import os

MIN_CHARS = 128  # จำนวนตัวอักษรขั้นต่ำต่อย่อหน้าที่รวมแล้ว

# 1. อ่านไฟล์ข้อความดิบ
with open("./dataset/ex-data-quality.txt", "r", encoding="utf-8") as f:
    raw_lines = [line.strip() for line in f if line.strip()]

# 2. กรองบรรทัดที่สั้นเกินไป (ชื่อ, อีเมล, URL — ไม่ใช่เนื้อหาจริง)
meaningful_lines = [line for line in raw_lines if len(line) >= 40]

# 3. รวมย่อหน้าสั้นๆ เข้าด้วยกัน ให้ได้ย่อหน้าที่มีเนื้อหาสมบูรณ์
grouped_paragraphs = []
current_group = ""
for line in meaningful_lines:
    if current_group:
        current_group += " " + line
    else:
        current_group = line

    if len(current_group) >= MIN_CHARS:
        grouped_paragraphs.append(current_group)
        current_group = ""

# ย่อหน้าสุดท้ายที่อาจสั้นกว่า MIN_CHARS → รวมกับย่อหน้าก่อนหน้า
if current_group:
    if grouped_paragraphs:
        grouped_paragraphs[-1] += " " + current_group
    else:
        grouped_paragraphs.append(current_group)

# 4. แบ่งข้อมูลเป็น Train (90%) และ Valid (10%)
split_idx = int(len(grouped_paragraphs) * 0.9)
if split_idx == 0:
    split_idx = 1  # อย่างน้อย Train ต้องมี 1 ตัวอย่าง
train_data = grouped_paragraphs[:split_idx]
valid_data = grouped_paragraphs[split_idx:]

# ถ้า valid ว่าง (ข้อมูลน้อยมาก) → ย้ายตัวสุดท้ายจาก train มา
if not valid_data and len(train_data) > 1:
    valid_data = [train_data.pop()]

print(f"📊 บรรทัดดิบทั้งหมด: {len(raw_lines)} บรรทัด")
print(f"📊 หลังกรอง (≥40 ตัวอักษร): {len(meaningful_lines)} บรรทัด")
print(f"📊 หลังรวมย่อหน้า (≥{MIN_CHARS} ตัวอักษร): {len(grouped_paragraphs)} ย่อหน้า")
print(f"📊 Train: {len(train_data)} ย่อหน้า")
print(f"📊 Valid: {len(valid_data)} ย่อหน้า")
print()
for i, text in enumerate(train_data):
    preview = text[:80] + "..." if len(text) > 80 else text
    print(f"   Train[{i}] ({len(text)} chars): {preview}")
for i, text in enumerate(valid_data):
    preview = text[:80] + "..." if len(text) > 80 else text
    print(f"   Valid[{i}] ({len(text)} chars): {preview}")
print()

# 5. บันทึกเป็น JSONL
def save_jsonl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for text in data:
            json_line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"💾 บันทึก: {filename} ({len(data)} ตัวอย่าง)")

save_jsonl(train_data, "./data/raw/train.jsonl")
save_jsonl(valid_data, "./data/raw/valid.jsonl")