"""
สคริปต์นี้แบ่งข้อมูลดิบเป็น Train/Valid สำหรับ Pretraining

📌 ค่า config ที่สำคัญ:
  • split_idx = int(len(paragraphs) * 0.8) → แบ่ง 80% Train, 20% Valid
    - ถ้า Train มาก: โมเดลเรียนรู้ข้อมูลได้เยอะ แต่ Validate น้อย
    - ถ้า Valid มาก: ประเมินผลแม่นยำ แต่ข้อมูลสอนน้อย
    - 80/20 เป็นค่ามาตรฐานที่ใช้กันทั่วไป

💡 ผลต่อ Inference:
  - ข้อมูล Train คือสิ่งที่โมเดล "จำ" ได้
  - ถ้าข้อมูลมีคุณภาพสูง → คำตอบแม่นยำ
  - ถ้าข้อมูลมีเสียงรบกวน → คำตอบผิดเพี้ยน
  - การแบ่งข้อมูลที่ดีช่วยให้โมเดลไม่ "จำเกิน" (overfit)
"""
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