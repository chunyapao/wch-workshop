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

  • dropout = 0.05 → โอกาส “ตัด” neuron ทิ้งตอนเทรน
    - ค่ามาก: ป้องกัน overfit ได้ดี แต่เรียนรู้ช้า
    - ค่าน้อย: เรียนรู้เร็ว แต่อาจ overfit
    - 0.05 (5%) เป็นค่าต่ำ เหมาะกับข้อมูลน้อย

💡 ผลต่อ Inference:
  - LoRA Adapters คือ “ความรู้ใหม่” ที่โมเดลเรียนเพิ่ม
  - ตอน inference ต้องโหลด adapter พร้อมโมเดล → คำตอบจะเปลี่ยนไป
  - rank/scale/dropout ที่เทรนไว้ ต้องตรงกับตอนโหลด adapter
  - ถ้าไม่ตรง → คำตอบจะผิดเพี้ยน หรือโหลดไม่ได้เลย
"""

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
    
    # 5. สร้างไฟล์ adapter_config.json สำหรับโหลด Adapter ในภายหลัง
    adapter_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters")
    os.makedirs(adapter_dir, exist_ok=True)
    
    adapter_config = {
        "model_type": model.model_type,
        "num_layers": num_lora_layers,
        "lora_parameters": {
            "rank": lora_config["rank"],
            "scale": lora_config["scale"],
            "dropout": lora_config["dropout"]
        }
    }
    
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=4, ensure_ascii=False)
    
    print(f"📄 บันทึก adapter_config.json ที่: {config_path}")
    print("✅ Model Initialization สำเร็จเรียบร้อยพร้อมเทรน!")
    return model, tokenizer

if __name__ == "__main__":
    initialize_model()