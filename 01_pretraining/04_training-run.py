"""
สคริปต์นี้รันการ Pretraining โมเดลด้วย LoRA

📌 ค่า config ที่สำคัญ:
  • iters=500 → จำนวนรอบการเทรน (อ่านจาก .env)
    - ถ้าค่าน้อย: เรียนรู้ไม่พอ คำตอบยังไม่ดี
    - ถ้าค่ามาก: เรียนรู้เยอะ แต่อาจ overfit (จำข้อมูลเทรนได้แต่ตอบคำถามใหม่ไม่ได้)
    - 500 steps เหมาะกับข้อมูลน้อยๆ

  • batch_size=1 → จำนวนตัวอย่างที่ประมวลผลพร้อมกัน (อ่านจาก .env)
    - ถ้าค่ามาก: เรียนรู้เร็ว ใช้ RAM มาก
    - ถ้าค่าน้อย: เรียนรู้ช้า แต่ใช้ RAM น้อย
    - 1 เหมาะกับ M1 8GB

  • max_seq_length=130 → ความยาวสูงสุดของข้อความที่โมเดลเห็นตอนเทรน
    - ต้องมากกว่า BLOCK_SIZE (128) เล็กน้อย
    - ถ้าตอน inference ป้อนข้อความยาวกว่านี้ → โมเดลจะ “ลืม” ส่วนที่เกิน

  • learning_rate=3e-5 → อัตราการเรียนรู้
    - ถ้าค่ามาก: เรียนรู้เร็ว แต่อาจ “ข้าม” จุดที่ดีที่สุด
    - ค่าน้อย: เรียนรู้ช้า แต่แม่นยำกว่า
    - 3e-5 (0.00003) เป็นค่ามาตรฐานสำหรับ LoRA

  • steps_per_eval=10 → ทดสอบกับ Validation Data ทุกๆ 10 steps
    - ค่ามาก: ประเมินผลน้อยครั้ง → ไม่รู้ว่า overfit เมื่อไหร่
    - ค่าน้อย: ประเมินผลบ่อย → ใช้เวลานาน

💡 ผลต่อ Inference:
  - iters ที่เหมาะสม → คำตอบแม่นยำ ไม่ overfit
  - max_seq_length → ความยาวบริบทที่โมเดล “จำ” ได้ตอน inference
  - learning_rate ที่เหมาะสม → คำตอบมีคุณภาพ ไม่ “เพี้ยน”
"""
import os
import types
from pathlib import Path
from dotenv import load_dotenv
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import CacheDataset

# โหลดค่าจากไฟล์ .env (ที่ root ของโปรเจกต์)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# อ่านค่าจาก .env
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "130"))
TRAINING_ITERS = int(os.getenv("TRAINING_ITERS", "500"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))

# การตั้งค่า LoRA
LORA_NUM_LAYERS = int(os.getenv("LORA_NUM_LAYERS", "8"))
LORA_RANK = int(os.getenv("LORA_RANK", "16"))
LORA_SCALE = float(os.getenv("LORA_SCALE", "2.0"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

def main():
    model_id = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
    data_dir = "./data/packed" # โฟลเดอร์ที่เก็บไฟล์ train.jsonl และ valid.jsonl (ข้อมูลไม่ห่อ)
    
    # -----------------------------------------
    # 1. โหลดโมเดล และ Tokenizer
    # -----------------------------------------
    print(f"⏳ กำลังโหลดโมเดล: {model_id}...")
    model, tokenizer = load(model_id)
    model.freeze()
    
    # -----------------------------------------
    # 2. ตั้งค่าและฝัง LoRA Adapters
    # -----------------------------------------
    lora_config = {
        "num_layers": LORA_NUM_LAYERS,
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": LORA_SCALE,
            "dropout": LORA_DROPOUT,
        },
    }
    
    print(f"⚙️ การตั้งค่า LoRA จาก .env:")
    print(f"   - num_layers: {LORA_NUM_LAYERS}")
    print(f"   - rank: {LORA_RANK}")
    print(f"   - scale: {LORA_SCALE}")
    print(f"   - dropout: {LORA_DROPOUT}")
    
    print("⚙️ กำลังฝัง LoRA Adapters...")
    linear_to_lora_layers(model, lora_config["num_layers"], config=lora_config["lora_parameters"])
    
    # เช็คพารามิเตอร์เพื่อความแน่ใจ
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"🎯 พารามิเตอร์ที่ต้องเทรน: {trainable/1e6:.2f}M ({trainable/total*100:.3f}%)\n")
    
    # -----------------------------------------
    # 3. โหลดชุดข้อมูล (Dataset)
    # -----------------------------------------
    # สร้าง Object จำลองเพื่อเก็บพาธของข้อมูลส่งให้ฟังก์ชัน load_dataset
    args = types.SimpleNamespace(data=data_dir, train=True, test=False)
    print(f"📁 กำลังโหลดข้อมูลจากโฟลเดอร์: {data_dir}/ ...")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    
    # ห่อ Dataset ใน CacheDataset เพื่อให้ iterate_batches สามารถเข้าถึงข้อมูลเป็น tuple (tokens, offset) ได้
    train_set = CacheDataset(train_set) if train_set else None
    valid_set = CacheDataset(valid_set) if valid_set else None
    
    # -----------------------------------------
    # 4. ตั้งค่า Hyperparameters สำหรับเทรน
    # -----------------------------------------
    print(f"📌 การตั้งค่าจาก .env:")
    print(f"   - MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
    print(f"   - TRAINING_ITERS: {TRAINING_ITERS}")
    print(f"   - LEARNING_RATE: {LEARNING_RATE}")
    print(f"   - BATCH_SIZE: {BATCH_SIZE}")
    
    training_args = TrainingArgs(
        adapter_file="adapters/adapters.safetensors", # ปลายทางเซฟโมเดล
        iters=TRAINING_ITERS,     # จำนวนรอบการเทรน (จาก .env)
        batch_size=BATCH_SIZE,        # จาก .env (1 สำหรับ RAM 8GB)
        max_seq_length=MAX_SEQ_LENGTH,  # ความยาวสูงสุด (จาก .env)
        steps_per_eval=10,        # ทดสอบกับ Validation Data ทุกๆ 10 Steps
        steps_per_save=100,       # เซฟ Checkpoint ทุกๆ 100 Steps
    )
    
    # ตั้งค่า Optimizer และ Learning Rate (จาก .env)
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    
    # -----------------------------------------
    # 5. ฟังก์ชันสำหรับรายงานผล (Metrics)
    # -----------------------------------------
    # คลาสนี้จะถูกเรียกใช้อัตโนมัติในลูป เพื่อให้เราเห็นตัวเลขขยับไปมาตอนเทรน
    class MetricsReporter:
        def on_train_loss_report(self, info):
            # แสดงผลทุกขั้นตอนการเทรน
            step = info['iteration']           # ขั้นตอนที่เท่าไหร่
            loss = info['train_loss']          # ค่าความผิดพลาด (ยิ่งน้อยยิ่งดี)
            speed = info['iterations_per_second']  # ความเร็วในการเทรน
            print(f"🔄 ขั้นตอนที่ {step:03d} | ความผิดพลาด: {loss:.4f} | ความเร็ว: {speed:.2f} ครั้ง/วินาที")
            
        def on_val_loss_report(self, info):
            # แสดงผลทดสอบ (ข้อมูลที่โมเดลไม่เคยเห็น)
            val_loss = info['val_loss']        # ค่าความผิดพลาดจากการทดสอบ
            print(f"\n📝 ผลทดสอบ (Validation): ความผิดพลาด = {val_loss:.4f}\n")
            
    metrics = MetricsReporter()
    
    # -----------------------------------------
    # 6. เริ่มเทรน!
    # -----------------------------------------
    print("\n🚀 เริ่มกระบวนการเทรนโมเดล...")
    os.makedirs("adapters", exist_ok=True) # สร้างโฟลเดอร์ adapters รอไว้
    
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=training_args,
        training_callback=metrics
    )
    
    print("\n✅ การเทรนเสร็จสมบูรณ์! ไฟล์ความรู้ใหม่ (Adapter) ถูกบันทึกไว้ที่: adapters/adapters.safetensors")

if __name__ == "__main__":
    main()