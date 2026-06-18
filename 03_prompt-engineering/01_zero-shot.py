"""
สคริปต์นี้แสดงการใช้งาน Zero-Shot Prompting (ถามคำถามโดยตรง)

📌 ค่า config ที่สำคัญ:
  • max_tokens=130 → จำกัดคำตอบ 130 Token
    - ถ้าค่าน้อย: คำตอบสั้น อาจไม่ครบถ้วน
    - ถ้าค่ามาก: คำตอบยาว แต่ใช้เวลานาน

💡 ผลต่อ Inference:
  - Zero-Shot = ถามคำถามโดยตรง ไม่ต้องมีตัวอย่าง
  - โมเดลใช้ความรู้ที่มีอยู่แล้วตอบคำถาม
  - ถ้าคำถามชัดเจน → คำตอบแม่นยำ
  - ถ้าคำถามกำกวม → คำตอบอาจไม่ตรงความคาดหวัง
"""
from mlx_lm import load, generate

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "../01_pretraining/adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = "บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?"

# หรือรูปแบบที่มีโครงสร้างชัดเจน
# 1. สร้าง Prompt สำหรับ Zero-Shot Prompting
prompt = """
คำถาม: บุญส่ง ศรีทอง ทำงานที่ odds  ทำงานที่แรกที่ไหน?
คำตอบ:
"""

# 2. เรียกใช้ generate() เพื่อสร้างคำตอบ

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=130,
    verbose=True
)

# สรุป Token Usage
prompt_tokens = len(tokenizer.encode(prompt))
response_tokens = len(tokenizer.encode(response))
print(f"\n{'='*40}")
print(f"Token Usage Summary")
print(f"{'='*40}")
print(f"Prompt tokens:   {prompt_tokens}")
print(f"Response tokens: {response_tokens}")
print(f"Total tokens:    {prompt_tokens + response_tokens}")
