import json
import os

# 1. อ่านไฟล์ข้อความดิบ
with open("./dataset/ex-data-quality.txt", "r", encoding="utf-8") as f:
    # แยกข้อความตามการขึ้นบรรทัดใหม่ (ย่อหน้า)
    paragraphs = [line.strip() for line in f if line.strip()]
    #

# 2. แบ่งข้อมูลเป็น Train (90%) และ Valid (10%)
split_idx = int(len(paragraphs) * 0.8)
train_data = paragraphs[:split_idx]
valid_data = paragraphs[split_idx:]

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