import subprocess
import sys
from pathlib import Path

# ใช้โมเดลพื้นฐานพร้อม adapter จาก 03_finetuning
model_id = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"

# ใช้ absolute path เพื่อให้ทำงานได้จากทุก directory
script_dir = Path(__file__).parent.resolve()
adapter_path = str(script_dir.parent / "03_finetuning" / "adapters")

cmd = [
    sys.executable, "-m", "mlx_lm.server",
    "--model", model_id,
    "--adapter-path", adapter_path,
]

print(f"Starting mlx_lm server...")
print(f"Model: {model_id}")
print(f"Adapter: {adapter_path}")
print(f"Server will be available at http://localhost:8080")
print("="*50)

subprocess.run(cmd)