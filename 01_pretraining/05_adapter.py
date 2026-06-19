"""
สคริปต์นี้แสดงการใช้งาน Base Model + Adapter ที่เทรนแล้ว เพื่อตอบคำถาม

📌 ค่า config ที่สำคัญ:
  • adapter_path = "./adapters" → โฟลเดอร์ที่เก็บ Adapter ที่เทรนไว้
    - ต้องมีไฟล์ adapters.safetensors และ adapter_config.json
    - ถ้าไม่มี → โหลดไม่ได้ หรือคำตอบไม่เปลี่ยนจาก Base Model

💡 ผลต่อ Inference:
  - โหลด Adapter → คำตอบเปลี่ยนไปจาก Base Model (มีความรู้ใหม่)
  - ไม่โหลด Adapter → คำตอบเป็นแบบ Base Model (ความรู้เดิม)
  - Adapter ที่เทรนมาดี → คำตอบแม่นยำขึ้น
  - Adapter ที่ overfit → คำตอบ “จำ” ข้อมูลเทรน แต่ตอบคำถามใหม่ไม่ได้
"""
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "./adapters"

# 📌 load() พร้อม adapter_path → โหลด Base Model + Adapter
# ผล: คำตอบจะเปลี่ยนไปจาก Base Model (มีความรู้ใหม่จาก Adapter)
model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = "บุญส่ง ศรีทอง เคยทำงานที่ไหน?"

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=256,
    verbose=True
)
