"""
สคริปต์นี้แสดงการใช้งาน Instruction Tuning ด้วย Chat Template

📌 ค่า config ที่สำคัญ:
  • apply_chat_template → แปลงข้อความเป็นรูปแบบ Chat
    - เพิ่ม Special Tokens เช่น <|begin_of_text|>, <|start_header_id|>
    - บอกโมเดลว่า “นี่คือคำถาม” และ “นี่คือคำตอบ”
    - สำคัญมากสำหรับโมเดลที่เทรนด้วย Chat Template

💡 ผลต่อ Inference:
  - ใช้ Chat Template → คำตอบเป็นรูปแบบบทสนทนา
  - ไม่ใช้ Chat Template → คำตอบอาจไม่ตรงรูปแบบที่โมเดลเรียนรู้
  - ถ้าตอนเทรนใช้ Chat Template → ตอน inference ต้องใช้ด้วย
"""
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "./adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = "บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?"

# 📌 apply_chat_template → แปลงข้อความเป็นรูปแบบ Chat
# ผล: โมเดลจะตอบเป็นรูปแบบบทสนทนา (ไม่ใช่ข้อความเพียวๆ)
messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

response = generate(
    model,
    tokenizer,
    prompt=prompt_text,
    max_tokens=130,
    verbose=True
)
