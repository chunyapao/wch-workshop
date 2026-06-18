"""
สคริปต์นี้เตรียมโมเดลสำหรับ Pretraining โดยฝัง LoRA Adapters

📌 ค่า config ที่สำคัญ:
  • num_lora_layers = 8 → จำนวนเลเยอร์ที่จะฝัง LoRA (นับจากท้าย)
    - ถ้าค่าน้อย: เรียนรู้ช้า แต่ประหยัด RAM
    - ถ้าค่ามาก: เรียนรู้เร็ว แต่ใช้ RAM มาก
    - 8 เลเยอร์สุดท้ายเป็นค่ามาตรฐานสำหรับโมเดล 1B

  • rank = 24 → ขนาดของ Matrix ใน LoRA
    - ถ้าค่าน้อย: ความสามารถจำกัด เรียนรู้ได้เฉพาะรูปแบบง่ายๆ
    - ถ้าค่ามาก: ความสามารถสูง แต่ใช้ RAM มากและอาจ overfit
    - 24 เป็นค่ากลางๆ สำหรับโมเดล 1B

  • scale = 2.0 → สเกลของ LoRA (คูณกับน้ำหนัก LoRA)
    - ค่ามาก: LoRA มีอิทธิพลมาก → คำตอบเปลี่ยนจาก Base Model มาก
    - ค่าน้อย: LoRA มีอิทธิพลน้อย → คำตอบใกล้เคียง Base Model
    - 2.0 เป็นค่าเริ่มต้นที่ใช้กันทั่วไป

  • dropout = 0.05 → โอกาส "ตัด" neuron ทิ้งตอนเทรน
    - ค่ามาก: ป้องกัน overfit ได้ดี แต่เรียนรู้ช้า
    - ค่าน้อย: เรียนรู้เร็ว แต่อาจ overfit
    - 0.05 (5%) เป็นค่าต่ำ เหมาะกับข้อมูลน้อย

💡 ผลต่อ Inference:
  - LoRA Adapters คือ "ความรู้ใหม่" ที่โมเดลเรียนเพิ่ม
  - ตอน inference ต้องโหลด adapter พร้อมโมเดล → คำตอบจะเปลี่ยนไป
  - rank/scale/dropout ที่เทรนไว้ ต้องตรงกับตอนโหลด adapter
  - ถ้าไม่ตรง → คำตอบจะผิดเพี้ยน หรือโหลดไม่ได้เลย
"""
import json
import os
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten


def initialize_model():
    """
    เตรียมโมเดลให้พร้อมเทรน
    
    ขั้นตอน:
    1. โหลด Base Model (Llama 3.2 แบบ 4-bit quantized)
    2. แช่แข็ง weights เดิมทั้งหมด (ไม่ให้เปลี่ยน)
    3. เพิ่ม LoRA adapters ที่ 8 เลเยอร์สุดท้าย
    4. บันทึก adapter_config.json สำหรับใช้ภายหลัง
    
    Returns:
        model: โมเดลที่พร้อมเทรน (มี LoRA attached)
        tokenizer: ตัว tokenizer สำหรับแปลงข้อความเป็น tokens
    """
    
    # ========================================================================
    # ขั้นตอนที่ 1: โหลด Base Model
    # ========================================================================
    # - ดึง model weights จาก HuggingFace Hub
    # - โมเดลเป็น 4-bit quantized (ลดขนาดจาก ~3GB เหลือ ~1GB)
    # - Tokenizer ใช้แปลงข้อความ <-> ตัวเลข
    model_id = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
    print(f"⏳ กำลังโหลด Base Model: {model_id}...")
    
    model, tokenizer = load(model_id)
    
    # ========================================================================
    # ขั้นตอนที่ 2: แช่แข็ง (Freeze) น้ำหนักดั้งเดิม
    # ========================================================================
    # - ล็อก weights ทั้งหมดของ Base Model ไม่ให้เปลี่ยน
    # - นี่คือหัวใจของ QLoRA: เทรนแค่ LoRA, ไม่แตะ Base Model
    # - ประหยัด RAM เพราะไม่ต้องเก็บ gradient ของ weights เดิม
    model.freeze()
    
    # ========================================================================
    # ขั้นตอนที่ 3: เพิ่ม LoRA Adapters
    # ========================================================================
    # - เพิ่ม "layer ย่อย" ที่สามารถเทรนได้ เข้าไปที่ Linear layers
    # - เลือก 8 เลเยอร์สุดท้าย (จาก 32 เลเยอร์) เพื่อประหยัด RAM
    # - LoRA จะเรียนรู้ "ส่วนต่าง" ของความรู้ใหม่ (ภาษาไทย)
    
    num_lora_layers = 8  # จำนวนเลเยอร์ที่จะเพิ่ม LoRA (นับจากท้าย)
    
    lora_config = {
        "rank": 24,      # ขนาด Matrix: ยิ่งใหญ่ = ความสามารถสูง แต่กิน RAM
        "scale": 2.0,    # อิทธิพลของ LoRA: 2.0 = LoRA มีน้ำหนัก 2 เท่า
        "dropout": 0.05  # ป้องกัน overfit: 5% = ตัด neuron ทิ้ง 5% ตอนเทรน
    }
    
    # เพิ่ม LoRA เข้าไปที่ Linear layers ของ 8 เลเยอร์สุดท้าย
    linear_to_lora_layers(model, num_lora_layers, config=lora_config)
    
    # ========================================================================
    # ขั้นตอนที่ 4: ตรวจสอบพารามิเตอร์ (Sanity Check)
    # ========================================================================
    # - นับจำนวน parameters ที่เทรนได้ (LoRA) vs ทั้งหมด
    # - ใช้ตรวจสอบว่า LoRA ถูกเพิ่มถูกต้อง
    
    # พารามิเตอร์ที่เทรนได้ (LoRA เท่านั้น)
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    
    # พารามิเตอร์ทั้งหมด (Base Model + LoRA)
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    
    print(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"📊 Total parameters: {total_params:,}")
    
    # ========================================================================
    # ขั้นตอนที่ 5: สร้าง adapter_config.json
    # ========================================================================
    # - บันทึกการตั้งค่า LoRA ไว้เป็นไฟล์ JSON
    # - จำเป็นสำหรับตอนโหลด adapter มาใช้ (inference)
    # - ถ้า config ไม่ตรงกับการเทรน → โหลดไม่ได้!
    
    adapter_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters")
    os.makedirs(adapter_dir, exist_ok=True)
    
    # สร้าง config dictionary
    adapter_config = {
        "model_type": model.model_type,      # "llama"
        "num_layers": num_lora_layers,       # 8
        "lora_parameters": {
            "rank": lora_config["rank"],     # 24
            "scale": lora_config["scale"],   # 2.0
            "dropout": lora_config["dropout"] # 0.05
        }
    }
    
    # บันทึกเป็นไฟล์ JSON
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=4, ensure_ascii=False)
    
    print(f"📄 บันทึก adapter_config.json ที่: {config_path}")
    print("✅ Model Initialization สำเร็จ! โมเดลพร้อมเทรนแล้ว")
    
    return model, tokenizer


# ============================================================================
# จุดเริ่มต้นของโปรแกรม
# ============================================================================
if __name__ == "__main__":
    # เรียกใช้ฟังก์ชัน initialize_model()
    # ได้ model ที่พร้อมเทรน + tokenizer
    model, tokenizer = initialize_model()
