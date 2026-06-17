import os
import json
import random
from datetime import datetime

def convert_to_isan_finetuning():
    source_files = [
        "../01_pretraining/data/raw/train.jsonl",
        "../01_pretraining/data/raw/valid.jsonl"
    ]
    
    all_texts = []
    for file_path in source_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("text", "").strip():
                    all_texts.append(data["text"].strip())
    
    print(f"พบข้อมูลทั้งหมด {len(all_texts)} รายการ")
    
    # พจนานุกรมแปลพื้นฐาน
    thai_to_isan = {
        "ไม่": "บ่",
        "เรา": "เฮา",
        "คุณ": "เจ้า",
        "ฉัน": "ข้อย",
        "ทำ": "เฮ็ด",
        "มาก": "หลาย",
        "ใช่": "แม่น",
        "ไม่ใช่": "บ่แม่น",
        "ไม่มี": "บ่มี",
        "ไม่ได้": "บ่ได้",
        "จะ": "สิ",
        "อะไร": "อี่หยัง",
        "ทำไม": "เพราะอี่หยัง",
        "อย่างไร": "จังใด๋",
        "ที่ไหน": "อยู่ใส",
        "ใคร": "ไผ",
        "ครับ": "เด้อ",
        "ค่ะ": "เด้อ",
        "นะ": "เด้อ",
    }
    
    formatted_data = []
    
    for text in all_texts:
        # แปลพื้นฐานโดยแทนที่คำ
        isan_text = text
        for thai_word, isan_word in thai_to_isan.items():
            isan_text = isan_text.replace(thai_word, isan_word)
        
        task_type = random.choice(["th_to_is", "question"])
        
        if task_type == "th_to_is":
            prompt = f"จงแปลประโยคต่อไปนี้เป็นภาษาอีสาน: {text}"
            completion = isan_text
        else:
            prompt = f"อธิบายเกี่ยวกับเรื่องนี้ในภาษาอีสาน: {text}"
            completion = f"เรื่องนี้เกี่ยวกับ {isan_text}"
        
        formatted_data.append({"prompt": prompt, "completion": completion})
    
    # โหลดข้อมูลเดิม
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
    
    # แบ่งข้อมูลใหม่ (90% train, 10% valid)
    train_data = formatted_data[:int(len(formatted_data)*0.9)]
    valid_data = formatted_data[int(len(formatted_data)*0.9):]
    
    # รวมกับข้อมูลเดิม
    final_train = existing_train + train_data
    final_valid = existing_valid + valid_data
    
    # บันทึก
    def save_jsonl(data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    save_jsonl(final_train, os.path.join(target_dir, "train.jsonl"))
    save_jsonl(final_valid, os.path.join(target_dir, "valid.jsonl"))
    
    print(f"บันทึกข้อมูล train: {len(final_train)} รายการ")
    print(f"บันทึกข้อมูล valid: {len(final_valid)} รายการ")

if __name__ == "__main__":
    convert_to_isan_finetuning()
