"""
สคริปต์นี้แสดงการใช้งาน Base Model (ก่อนเทรน) เพื่อตอบคำถาม

📌 ค่า config ที่สำคัญ:
  • max_tokens=128 → จำกัดคำตอบ 128 Token
    - ถ้าค่าน้อย: คำตอบสั้น อาจไม่ครบถ้วน
    - ถ้าค่ามาก: คำตอบยาว แต่อาจพูดน้ำท่วม (hallucinate)
    - 128 Token ≈ 60-80 คำภาษาไทย เหมาะสำหรับการตอบคำถามสั้นๆ

  • verbose=True → แสดง Token ที่กำลังสร้างแบบ Real-time
    - ไม่มีผลต่อคุณภาพคำตอบ แค่ทำให้เห็นว่าโมเดลกำลังทำอะไร

💡 ผลของ 4-bit Quantization ต่อ Inference:
  - Base Model ถูกบีบอัดจาก 16-bit → 4-bit
  - ใช้ RAM ≈ 1GB แทน 4GB → รันบน M1 8GB ได้สบาย
  - ความแม่นยำลดลงเล็กน้อย แต่พอใช้ได้ในงานทั่วไป
"""
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")
prompt = "บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?"

# 📌 generate() = สร้างคำตอบจากโมเดล
# max_tokens=128 → คำตอบยาวสุด 128 Token (≈60-80 คำ)
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=128,
    verbose=True
)
