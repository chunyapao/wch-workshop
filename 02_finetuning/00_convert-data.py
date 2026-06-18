"""
สคริปต์นี้แปลงข้อมูลจาก Dataset ภาษาอีสานเป็นรูปแบบ Prompt-Completion สำหรับ Fine-tuning

📌 ค่า config ที่สำคัญ:
  • dataset_name = "typhoon-ai/thai-dialect-isan-dataset" → Dataset จาก Hugging Face
    - มีข้อมูลภาษาไทย → ภาษาอีสานระดับประโยค
    - ครอบคลุมกว่าพจนานุกรมแบบคำต่อคำ
    - มีทั้งคำถามและคำตอบที่สมบูรณ์

  • task_type = random.choice(["th_to_is", "question"]) → สุ่มประเภทงาน
    - th_to_is: สอนให้แปลภาษาไทย → ภาษาอีสาน
    - question: สอนให้ตอบคำถามเป็นภาษาอีสาน
    - การมีหลายประเภทช่วยให้โมเดล generalize ได้ดี

💡 ผลต่อ Inference:
  - ข้อมูลจาก Dataset มีคุณภาพสูงกว่าการแปลแบบคำต่อคำ
  - โมเดลจะเรียนรู้รูปแบบการแปลระดับประโยค
  - ถ้า Dataset มีคุณภาพสูง → คำตอบแม่นยำ
  - การมีหลายรูปแบบ (แปล/ตอบคำถาม) ช่วยให้โมเดลยืดหยุ่น

🔧 การแบ่งข้อมูล:
  • MAX_SAMPLES = 1000 → จำกัดจำนวนข้อมูลจาก Dataset (ป้องกันข้อมูลเยอะเกินไป)
    - ถ้าค่าน้อย: เทรนเร็ว แต่อาจไม่ครอบคลุม
    - ถ้าค่ามาก: เทรนช้า แต่ครอบคลุมมากขึ้น
    - ปรับค่าได้ตามความเหมาะสม (เช่น 500, 1000, 2000)
"""
import os
import json
import random
from datetime import datetime
from datasets import load_dataset

def convert_to_isan_finetuning():
    dataset_name = "typhoon-ai/thai-dialect-isan-dataset"
    MAX_SAMPLES = 1000  # จำกัดจำนวนข้อมูลจาก Dataset (ปรับได้ตามต้องการ)
    
    # 1. โหลด Dataset จาก Hugging Face แบบ Streaming (เร็วขึ้น ไม่ต้องโหลดทั้งหมด)
    print(f"⏳ กำลังโหลด Dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    formatted_data = []
    skipped = 0
    total = 0
    
    # 2. แปลงข้อมูลเป็นรูปแบบ Prompt-Completion (จำกัดจำนวนด้วย MAX_SAMPLES)
    print(f"⏳ กำลังแปลงข้อมูล... (จำกัด {MAX_SAMPLES} รายการ)")
    for row in dataset:
        if total >= MAX_SAMPLES:
            break
        total += 1
        
        question = row.get("question", "")
        isan_spelling = row.get("isan_spelling", "")
        thai_spelling = row.get("thai_spelling", "")
        
        # ข้ามข้อมูลที่มีคำตอบว่างเปล่า
        if not isan_spelling or str(isan_spelling).strip() == "":
            skipped += 1
            continue
        
        # สุ่มประเภทงาน
        task_type = random.choice(["th_to_is", "question"])
        
        # รูปแบบที่ 1: สอนให้แปลภาษาไทย → ภาษาอีสาน
        if task_type == "th_to_is" and thai_spelling and str(thai_spelling).strip() != "":
            prompt = f"จงแปลประโยคต่อไปนี้เป็นภาษาอีสาน: {thai_spelling.strip()}"
            completion = isan_spelling.strip()
            formatted_data.append({"prompt": prompt, "completion": completion})
        
        # รูปแบบที่ 2: สอนให้ตอบคำถามเป็นภาษาอีสาน
        elif question and str(question).strip() != "":
            prompt = f"อธิบายเกี่ยวกับเรื่องนี้ในภาษาอีสาน: {question.strip()}"
            completion = isan_spelling.strip()
            formatted_data.append({"prompt": prompt, "completion": completion})
        else:
            skipped += 1
    
    print(f"\n📊 ประมวลผลทั้งหมด {total} รายการ")
    print(f"📊 สร้างข้อมูลสำหรับ Fine-tuning ได้ทั้งหมด {len(formatted_data)} รายการ (ข้าม {skipped} รายการ)")
    
    # 3. โหลดข้อมูลเดิมจาก Pretraining (ถ้ามี)
    target_dir = "./data/raw"
    
    existing_train = []
    existing_valid = []
    
    for file_name in ["train.jsonl", "valid.jsonl"]:
        file_path = os.path.join(target_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
                if file_name == "train.jsonl":
                    existing_train = data
                else:
                    existing_valid = data
    
    print(f"📁 พบข้อมูลเดิมจาก Pretraining: Train {len(existing_train)} รายการ, Valid {len(existing_valid)} รายการ")
    
    # 4. แบ่งข้อมูลใหม่ (90% train, 10% valid)
    train_data = formatted_data[:int(len(formatted_data)*0.9)]
    valid_data = formatted_data[int(len(formatted_data)*0.9):]
    
    # 5. รวมกับข้อมูลเดิม
    final_train = existing_train + train_data
    final_valid = existing_valid + valid_data
    
    # 6. บันทึกข้อมูลลงไฟล์
    def save_jsonl(data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    os.makedirs(target_dir, exist_ok=True)
    save_jsonl(final_train, os.path.join(target_dir, "train.jsonl"))
    save_jsonl(final_valid, os.path.join(target_dir, "valid.jsonl"))
    
    print(f"\n✅ บันทึกข้อมูลเรียบร้อย!")
    print(f"📚 ข้อมูล Train: {len(final_train)} รายการ")
    print(f"📚 ข้อมูล Valid: {len(final_valid)} รายการ")

if __name__ == "__main__":
    convert_to_isan_finetuning()
