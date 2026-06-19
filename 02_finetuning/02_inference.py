"""
สคริปต์นี้ทดสอบคำตอบของโมเดลหลัง Fine-tuning ด้วย Chat Template

📌 ค่า config ที่สำคัญ:
  • adapter_path = "./adapters" → ใช้ Adapter ที่เทรนจาก 01_training-run-isan.py
  • apply_chat_template → แปลงข้อความเป็นรูปแบบ Chat
    - เพิ่ม Special Tokens เช่น <|begin_of_text|>, <|start_header_id|>
    - บอกโมเดลว่า "นี่คือคำถาม" และ "นี่คือคำตอบ"
    - สำคัญมากสำหรับโมเดลที่เทรนด้วย Chat Template
  • max_tokens=768 → จำกัดความยาวคำตอบ 768 tokens

💡 ผลต่อ Inference:
  - ใช้ Chat Template → คำตอบเป็นรูปแบบบทสนทนา
  - ไม่ใช้ Chat Template → คำตอบอาจไม่ตรงรูปแบบที่โมเดลเรียนรู้
  - ถ้าตอนเทรนใช้ Chat Template → ตอน inference ต้องใช้ด้วย

⚠️ ข้อจำกัด:
  - ไม่มี System Prompt บังคับภาษาอีสาน → คำตอบอาจเป็นไทยมาตรฐาน
  - ข้อมูลเทรนมีน้อย (4 รายการ) → โมเดลจำไม่ได้ทั้งประโยค
  - ถ้าต้องการคำตอบอีสานแท้ → เพิ่ม System Prompt หรือเพิ่มข้อมูลเทรน
"""
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "./adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = "บุญส่ง ศรีทอง เคยทำงานที่ไหน?"

# 📌 apply_chat_template → แปลงข้อความเป็นรูปแบบ Chat
# ผล: โมเดลจะตอบเป็นรูปแบบบทสนทนา (ไม่ใช่ข้อความเพียวๆ)
# ⚠️ ไม่มี System Prompt → คำตอบอาจไม่อีสาน (เพิ่ม system role ได้ถ้าต้องการ)
messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

response = generate(
    model,
    tokenizer,
    prompt=prompt_text,
    max_tokens=768,
    verbose=True
)
