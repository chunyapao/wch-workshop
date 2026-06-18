"""
สคริปต์นี้รันการ Pretraining โมเดลด้วย LoRA

📌 ค่า config ที่สำคัญ:
  • iters=100 → จำนวนรอบการเทรน
    - ถ้าค่าน้อย: เรียนรู้ไม่พอ คำตอบยังไม่ดี
    - ถ้าค่ามาก: เรียนรู้เยอะ แต่อาจ overfit (จำข้อมูลเทรนได้แต่ตอบคำถามใหม่ไม่ได้)
    - 100 steps เหมาะกับข้อมูลน้อยๆ

  • batch_size=1 → จำนวนตัวอย่างที่ประมวลผลพร้อมกัน
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
        "num_layers": 8,
        "lora_parameters": {
            "rank": 16,
            "scale": 2.0,   # ใช้ scale สำหรับ mlx-lm เวอร์ชันใหม่
            "dropout": 0.05,
        },
    }
    
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
    training_args = TrainingArgs(
        adapter_file="adapters/adapters.safetensors", # ปลายทางเซฟโมเดล
        iters=100,               # เทรนทั้งหมด 100 Steps
        batch_size=1,            # 1 สำหรับ RAM 8GB
        max_seq_length=130,      # ตัดข้อความที่ 130 tokens (รองรับความยาวสูงสุดของข้อมูลที่ Pack ไว้)
        steps_per_eval=10,       # ทดสอบกับ Validation Data ทุกๆ 50 Steps
        steps_per_save=100,      # เซฟ Checkpoint กันเหนียวทุกๆ 100 Steps
    )
    
    # ตั้งค่า Optimizer และ Learning Rate
    optimizer = optim.Adam(learning_rate=3e-5)
    
    # -----------------------------------------
    # 5. ฟังก์ชันสำหรับรายงานผล (Metrics)
    # -----------------------------------------
    # คลาสนี้จะถูกเรียกใช้อัตโนมัติในลูป เพื่อให้เราเห็นตัวเลขขยับไปมาตอนเทรน
    class MetricsReporter:
        def on_train_loss_report(self, info):
            print(f"🔄 Step {info['iteration']:03d} | Train Loss: {info['train_loss']:.4f} | ความเร็ว: {info['iterations_per_second']:.2f} iters/sec")
            
        def on_val_loss_report(self, info):
            print(f"\n✨ ---> Validation Loss: {info['val_loss']:.4f} <---✨\n")
            
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