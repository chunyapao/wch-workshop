"""
สคริปต์นี้แสดงการทำงานของ Tokenizer ซึ่งเป็นขั้นตอนแรกสุดก่อนที่โมเดลจะประมวลผล

📌 ค่า config ที่สำคัญ:
  • max_tokens=20 → จำกัดจำนวน Token ที่โมเดลจะสร้างได้
    - ถ้าค่าน้อย: คำตอบสั้น ถูกตัดกลางคัน (ประหยัดเวลา)
    - ถ้าค่ามาก: คำตอบยาวขึ้น แต่ใช้เวลานานและอาจ "น้ำท่วม" (hallucinate)
    - ตอน inference: ค่านี้ควบคุมความยาวสูงสุดของคำตอบโดยตรง

  • clean_up_tokenization_spaces=False → ป้องกันไม่ให้ Tokenizer ลบช่องว่างอัตโนมัติ
    - ถ้าไม่ตั้ง: คำตอบอาจมีช่องว่างหายไป เช่น "บ้าน ใหญ่" → "บ้านใหญ่"
    - สำคัญมากเมื่อต้องการรักษาโครงสร้างข้อความเดิม
"""
from mlx_lm import load, generate


# 📌 โมเดล 4-bit = โมเดลถูกบีบอัดจาก 16-bit เหลือ 4-bit
# ผลต่อ inference: ใช้ RAM น้อยลง (~1GB แทน ~4GB) แต่ความแม่นยำลดลงเล็กน้อย
model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")

prompt = "รู้ไหมว่าชื่อบุญส่งไหม?"

# Encode: แปลงข้อความ → ลำดับตัวเลข (Token IDs)
# โมเดลเข้าใจแต่ตัวเลขเท่านั้น ไม่เข้าใจตัวอักษร
input_ids = tokenizer.encode(prompt)

# Decode: แปลงตัวเลขกลับเป็นตัวอักษร (ทีละตัว)
for i, token_id in enumerate(input_ids):
    token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    print(f"  [{i}] {token_id} → {repr(token_text)}")

# 📌 generate() = ฟังก์ชันหลักที่สร้างคำตอบ
# max_tokens=20 → โมเดลสร้างได้มากสุด 130 Token (≈10-15 คำภาษาไทย)
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=130,
)

print()
print(f"Prompt Tokens: {len(input_ids)}")
print(f"Prompt Characters: {len(prompt)}")

