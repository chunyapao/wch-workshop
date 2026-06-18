"""
สคริปต์นี้ “จัดแพ็ก” ข้อมูลดิบให้เป็นก้อนๆ ขนาดเท่ากัน เพื่อเตรียมเทรน

📌 ค่า config ที่สำคัญ:
  • BLOCK_SIZE = 128 → ขนาดของแต่ละก้อนข้อมูล (Token)
    - ถ้าค่าน้อย: ข้อมูลถูกหั่นละเอียด โมเดลเห็นบริบทสั้น
    - ถ้าค่ามาก: ข้อมูลแต่ละก้อนยาว โมเดลเห็นบริบทยาว แต่ใช้ RAM มาก
    - 128 Token ≈ 60-80 คำ เหมาะกับข้อมูลภาษาไทย

  • add_special_tokens=False → ไม่เติม <bos>/<eos> อัตโนมัติ
    - เราเติมเองเพื่อให้ควบคุมได้ว่าก้อนไหนเริ่ม-จบ

💡 ผลต่อ Inference:
  - BLOCK_SIZE = max_seq_length ตอนเทรน = ความยาวบริบทที่โมเดล “จำ” ได้
  - ถ้าตอน inference ป้อนข้อความยาวกว่า 128 Token → โมเดลจะ “ลืม” ส่วนที่เกิน
  - bos/eos ช่วยบอกโมเดลว่า “ข้อความเริ่มที่นี่” และ “จบที่นี่”
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer

# โหลดค่าจากไฟล์ .env (ที่ root ของโปรเจกต์)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "128"))


def pack_data(input_file, output_file, tokenizer_name, block_size=512):
    print(f"กำลังเตรียมข้อมูลจาก {input_file}...")

    # 1. โหลด Tokenizer ของ Typhoon (โหลดเฉพาะ Tokenizer ไม่กินสเปค)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, clean_up_tokenization_spaces=False)

    all_tokens = []

    # 2. อ่านไฟล์ jsonl เดิม
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']

            # แปลงข้อความเป็น Token IDs (ไม่เติม Special Token อัตโนมัติ เพื่อคุมเอง)
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # แปะ <bos> (Begin of Sentence) และ <eos> (End of Sentence) คร่อมข้อความแต่ละชุด
            tokens = [tokenizer.bos_token_id] + \
                tokens + [tokenizer.eos_token_id]
            all_tokens.extend(tokens)

    # 3. หั่น Token ทั้งหมดเป็นก้อนๆ (Chunks)
    # เก็บทุกก้อน แม้ก้อนสุดท้ายจะไม่ครบ block_size (เพื่อไม่เสียข้อมูล)
    packed_data = []
    for i in range(0, len(all_tokens), block_size):
        chunk = all_tokens[i: i + block_size]

        # เก็บทุกก้อนที่มีข้อมูล (ก้อนสุดท้ายอาจสั้นกว่า block_size)
        if len(chunk) > 0:
            # แปลงกลับเป็น Text เพื่อให้ mlx_lm นำไปอ่านได้แบบไม่มีปัญหาเรื่อง Format
            chunk_text = tokenizer.decode(chunk)
            packed_data.append({"text": chunk_text})

    # 4. บันทึกลงไฟล์ jsonl ใหม่
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in packed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(
        f"✅ จัดแพ็กเสร็จสิ้น! ได้ข้อมูลทั้งหมด {len(packed_data)} Blocks (ขนาด Block ละ {block_size} tokens)")
    print(f"💾 บันทึกไฟล์ที่: {output_file}\n")


if __name__ == "__main__":
    # ระบุชื่อโมเดลของ Typhoon เพื่อใช้ Tokenizer ให้ตรงตัว
    TOKENIZER_ID = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
    
    print(f"📌 BLOCK_SIZE จาก .env: {BLOCK_SIZE}")

    # สมมติว่าไฟล์เดิมชื่อ train.jsonl และ valid.jsonl อยู่ในโฟลเดอร์ data/
    pack_data(
        input_file="data/raw/train.jsonl",
        output_file="./data/packed/train.jsonl",
        tokenizer_name=TOKENIZER_ID,
        block_size=BLOCK_SIZE
    )

    pack_data(
        input_file="./data/raw/valid.jsonl",
        output_file="./data/packed/valid.jsonl",
        tokenizer_name=TOKENIZER_ID,
        block_size=BLOCK_SIZE
    )
