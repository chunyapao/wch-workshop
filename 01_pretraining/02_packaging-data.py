import json
from transformers import AutoTokenizer


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

    # 3. หั่น Token ทั้งหมดเป็นก้อนๆ (Chunks) ขนาดเท่ากับ block_size เป๊ะๆ
    packed_data = []
    for i in range(0, len(all_tokens), block_size):
        chunk = all_tokens[i: i + block_size]

        # ตัดเศษตอนท้ายทิ้งถ้าความยาวไม่ถึง block_size
        if len(chunk) == block_size:
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
    BLOCK_SIZE = 128  # ปรับให้ตรงกับ --max-seq-length ใน mlx_lm

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
