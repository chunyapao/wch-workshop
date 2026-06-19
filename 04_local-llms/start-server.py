"""
สคริปต์นี้เริ่ม Local LLM Server สำหรับให้บริการ API

📌 ค่า config ที่สำคัญ:
  • model_id = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit" → โมเดลพื้นฐาน
    - ต้องเป็นโมเดลที่มี config.json
    - ถ้าไม่มี → โหลดไม่ได้

  • adapter_path → โฟลเดอร์ที่เก็บ Adapter ที่เทรนไว้
    - ต้องมีไฟล์ adapters.safetensors และ adapter_config.json
    - ถ้าไม่มี → โหลดไม่ได้ หรือคำตอบไม่เปลี่ยนจาก Base Model

💡 ผลต่อ Inference:
  - Server ให้บริการ API แบบ OpenAI-compatible
  - Client สามารถส่งคำขอผ่าน HTTP POST
  - Adapter ที่โหลด → คำตอบเปลี่ยนไปจาก Base Model
  - ถ้า Adapter ดี → คำตอบแม่นยำ
  - ถ้า Adapter overfit → คำตอบ “จำ” ข้อมูลเทรน แต่ตอบคำถามใหม่ไม่ได้
"""
import subprocess
import sys
from pathlib import Path

# 1. กำหนดค่าโมเดลและ Adapter
model_id = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"

# ใช้ absolute path เพื่อให้ทำงานได้จากทุก directory
script_dir = Path(__file__).parent.resolve()
adapter_path = str(script_dir.parent / "02_finetuning" / "adapters")

# 2. สร้างคำสั่งสำหรับเริ่ม Server
cmd = [
    sys.executable, "-m", "mlx_lm.server",
    "--model", model_id,
    "--adapter-path", adapter_path,
]

# 3. เริ่ม Server
print(f"Starting mlx_lm server...")
print(f"Model: {model_id}")
print(f"Adapter: {adapter_path}")
print(f"Server will be available at http://localhost:8080")
print("="*50)

subprocess.run(cmd)