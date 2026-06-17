import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten  # <-- เพิ่มบรรทัดนี้

def initialize_model():
    model_id = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
    print(f"⏳ กำลังโหลด Base Model: {model_id}...")
    
    # 1. โหลดโมเดล 4-bit และ Tokenizer 
    # (ขั้นตอนนี้จะดึง Weights เข้าสู่ RAM แบบบีบอัด)
    model, tokenizer = load(model_id)
    
    # 2. แช่แข็ง (Freeze) น้ำหนักดั้งเดิมทั้งหมด
    # เพื่อป้องกันไม่ให้ Base Model ถูกอัปเดต (หัวใจหลักของ QLoRA)
    model.freeze()
    
    # 3. ตั้งค่าและฝัง LoRA Adapters
    # เลือกจำนวนเลเยอร์ที่จะฝัง (เช่น 8 เลเยอร์สุดท้าย) และตั้งค่า Rank
    num_lora_layers = 8
    lora_config = {
        "rank": 24,     # ขนาดของ Matrix ยิ่งเยอะยิ่งฉลาดแต่กิน RAM
        "scale": 2.0,   # สเกลของ LoRA (คำนวณจาก alpha / rank -> เช่น 32 / 16 = 2.0)
        "dropout": 0.05
    }
    
    linear_to_lora_layers(model, num_lora_layers, config=lora_config)
    
    # 4. คำนวณเพื่อตรวจสอบพารามิเตอร์ (Sanity Check)
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    
    print("✅ Model Initialization สำเร็จเรียบร้อยพร้อมเทรน!")
    return model, tokenizer

if __name__ == "__main__":
    initialize_model()