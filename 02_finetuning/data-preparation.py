"""
สคริปต์นี้โหลด Dataset ภาษาอีสานจาก Hugging Face และเตรียมข้อมูลสำหรับ Fine-tuning

📌 ค่า config ที่สำคัญ:
  • test_size=0.1 → แบ่ง 10% สำหรับ Validation
    - ถ้าค่ามาก: ข้อมูลเทรนน้อย แต่ Validate แม่นยำ
    - ค่าน้อย: ข้อมูลเทรนเยอะ แต่ Validate น้อย
    - 10% เป็นค่ามาตรฐาน

💡 ผลต่อ Inference:
  - ข้อมูลจาก Dataset คือสิ่งที่โมเดลจะเรียนรู้
  - ถ้า Dataset มีคุณภาพสูง → คำตอบแม่นยำ
  - ถ้า Dataset มีเสียงรบกวน → คำตอบผิดเพี้ยน
  - การแบ่งข้อมูลที่ดีช่วยให้โมเดลไม่ “จำเกิน” (overfit)
"""
import os
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def prepare_isan_persona_data():
    dataset_name = "typhoon-ai/thai-dialect-isan-dataset"
    
    # โหลด Dataset แบบ Memory-map (ไม่ดึงไฟล์เสียงขึ้นมาบน RAM จนกว่าจะเรียกใช้)
    dataset = load_dataset(dataset_name, split="train")
    
    formatted_data = []
    
    # วนลูปอ่านข้อมูลทีละบรรทัดเพื่อเซฟ RAM
    for row in dataset:
        question = row.get("question", "")
        isan_spelling = row.get("isan_spelling", "")
        thai_spelling = row.get("thai_spelling", "")
        
        # ป้องกันกรณีข้อมูลคำตอบว่างเปล่า
        if not isan_spelling or str(isan_spelling).strip() == "":
            continue
            
        # ✨ รูปแบบที่ 1: ฝังบุคลิก (Persona) ให้ตอบคำถามเป็นภาษาอีสานเสมอ
        # สังเกตว่าเราป้อนแค่คำถามเพียวๆ โดยไม่มีคำสั่งกำกับ
        if question and str(question).strip() != "":
            formatted_data.append({
                "prompt": f"คำถาม: {question.strip()}",
                "completion": isan_spelling.strip()
            })
        
        # ✨ รูปแบบที่ 2: สอนให้แปลภาษา (Translator) เผื่อผู้ใช้สั่งแปลตรงๆ
        if thai_spelling and str(thai_spelling).strip() != "" and thai_spelling != isan_spelling:
            formatted_data.append({
                "prompt": f"จงแปลประโยคนี้เป็นภาษาอีสาน: {thai_spelling.strip()}",
                "completion": isan_spelling.strip()
            })

    
    # แบ่งข้อมูลเป็น Train (90%) สำหรับสอน และ Valid (10%) สำหรับประเมินผล
    train_data, valid_data = train_test_split(formatted_data, test_size=0.1, random_state=42)
    
    # สร้างโครงสร้างโฟลเดอร์สำหรับเก็บไฟล์
    os.makedirs("data/raw", exist_ok=True)
    
    # ฟังก์ชันบันทึกไฟล์เป็น JSONL
    def save_jsonl(data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    save_jsonl(train_data, "./data/raw/train.jsonl")
    save_jsonl(valid_data, "./data/raw/valid.jsonl")
    

if __name__ == "__main__":
    prepare_isan_persona_data()