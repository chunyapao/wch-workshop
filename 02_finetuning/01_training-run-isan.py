"""
สคริปต์นี้รันการ Fine-tuning โมเดลด้วย Adapter จาก Pretraining แบบ 2 รอบ

🔄 รอบที่ 1: เรียนรู้ภาษาอีสานจาก data/dataset/ (ข้อมูลใหญ่ ~9.5MB)
  - สอนให้โมเดลเข้าใจและตอบเป็นภาษาอีสาน
  - ข้อมูลจาก Dataset ภาษาอีสานระดับประโยค

🔄 รอบที่ 2: เรียนรู้ข้อมูลบุญส่งเป็นภาษาอีสานจาก data/raw/ (ข้อมูลเล็ก)
  - สอนให้โมเดลตอบคำถามเกี่ยวกับบุญส่งเป็นภาษาอีสาน
  - ข้อมูลประวัติการทำงาน, ความเชี่ยวชาญ, ข้อมูลส่วนตัว

📌 ค่า config ที่สำคัญ:
  • adapter_path = "../01_pretraining/adapters" → โฟลเดอร์ที่เก็บ Adapter จาก Pretraining
    - ต้องมีไฟล์ adapters.safetensors และ adapter_config.json
    - ถ้าไม่มี → โหลดไม่ได้ หรือคำตอบไม่เปลี่ยนจาก Base Model

  • max_seq_length=768 → ความยาวสูงสุดของข้อความที่โมเดลเห็นตอนเทรน
    - ถ้าค่ามาก: โมเดลเห็นบริบทยาว แต่ใช้ RAM มาก
    - ค่าน้อย: โมเดลเห็นบริบทสั้น แต่ใช้ RAM น้อย
    - 768 รองรับข้อมูลยาวสูงสุด ~583 tokens ในรอบที่ 2

  • learning_rate=1e-4 → อัตราการเรียนรู้
    - ถ้าค่ามาก: เรียนรู้เร็ว แต่อาจ "ข้าม" จุดที่ดีที่สุด
    - ค่าน้อย: เรียนรู้ช้า แต่แม่นยำกว่า
    - 1e-4 (0.0001) เป็นค่ามาตรฐานสำหรับ Fine-tuning

  • prompt_feature="prompt", completion_feature="completion"
    - บอกโมเดลว่า "คำถาม" อยู่ที่ไหน และ "คำตอบ" อยู่ที่ไหน
    - โมเดลจะเรียนรู้เฉพาะส่วน completion (ไม่เรียนรู้ส่วน prompt)
    - สำคัญมากสำหรับ Instruction Tuning

💡 ผลต่อ Inference:
  - Adapter ที่เทรนแล้ว → คำตอบเปลี่ยนไป (มีความรู้ใหม่)
  - max_seq_length → ความยาวบริบทที่โมเดล "จำ" ได้ตอน inference
  - prompt/completion format → โมเดลเรียนรู้รูปแบบเฉพาะ ถ้าตอน inference ใช้รูปแบบเดียวกัน → คำตอบแม่นยำ

⚠️ ข้อจำกัดปัจจุบัน:
  - ข้อมูลรอบที่ 2 มีน้อย (4 รายการ) → โมเดลอาจจำไม่ได้ทั้งประโยค
  - ข้อมูลไม่อีสานแท้ → โครงสร้างประโยคยังเป็นไทยมาตรฐาน
  - ถ้าต้องการอีสานแท้ → เพิ่มข้อมูลเทรนหรือเขียน completion ด้วยมือ
"""
import os
import json
import shutil
import types
import mlx.optimizers as optim
import mlx.utils
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, train
from mlx_lm.tuner.utils import load_adapters
from mlx_lm.tuner.datasets import load_dataset, CacheDataset

def run_training_phase(model, tokenizer, data_dir, adapter_file, phase_name, iters, learning_rate=1e-4):
    """รันการเทรนแต่ละรอบ"""
    print(f"\n{'='*60}")
    print(f"🔄 {phase_name}")
    print(f"📂 ข้อมูล: {data_dir}")
    print(f"💾 Adapter: {adapter_file}")
    print(f"{'='*60}")
    
    args = types.SimpleNamespace(
        data=data_dir,
        train=True,
        test=False,
        prompt_feature="prompt",
        completion_feature="completion"
    )
    
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    
    training_args = TrainingArgs(
        adapter_file=adapter_file,
        iters=iters,
        batch_size=1,
        max_seq_length=768,
        steps_per_eval=10,
        steps_per_save=iters,
    )
    
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    class MetricsReporter:
        def on_train_loss_report(self, info):
            # แสดงผลทุกขั้นตอนการเทรน
            step = info['iteration']           # ขั้นตอนที่เท่าไหร่
            loss = info['train_loss']          # ค่าความผิดพลาด (ยิ่งน้อยยิ่งดี)
            speed = info['iterations_per_second']  # ความเร็วในการเทรน
            print(f"🔄 ขั้นตอนที่ {step:04d} | ความผิดพลาด: {loss:.4f} | ความเร็ว: {speed:.2f} ครั้ง/วินาที")
            
        def on_val_loss_report(self, info):
            # แสดงผลทดสอบ (ข้อมูลที่โมเดลไม่เคยเห็น)
            val_loss = info['val_loss']        # ค่าความผิดพลาดจากการทดสอบ
            print(f"\n📝 ผลทดสอบ (Validation): ความผิดพลาด = {val_loss:.4f}\n")
    
    os.makedirs(os.path.dirname(adapter_file), exist_ok=True)
    
    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=MetricsReporter()
    )

def main():
    model_id = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
    pretraining_adapter_path = "../01_pretraining/adapters"
    
    # ================================================================
    # โหลดโมเดล + Adapter จาก Pretraining
    # ================================================================
    model, tokenizer = load(model_id)
    model.freeze()
    model = load_adapters(model, pretraining_adapter_path)
    
    trainable_params = mlx.utils.tree_flatten(model.trainable_parameters())
    print(f"✅ มีพารามิเตอร์สำหรับเทรน: {len(trainable_params)} รายการ")
    
    # ================================================================
    # รอบที่ 1: เรียนรู้ภาษาอีสานจาก data/dataset/ (ข้อมูลใหญ่)
    # ================================================================
    run_training_phase(
        model=model,
        tokenizer=tokenizer,
        data_dir="./data/dataset",
        adapter_file="./adapters/phase1/adapters.safetensors",
        phase_name="รอบที่ 1: เรียนรู้ภาษาอีสาน (ข้อมูลใหญ่)",
        iters=100
    )
    
    # สำเนา adapter_config.json ไปยัง phase1/ (mlx_lm train() ไม่ได้สร้างให้อัตโนมัติ)
    src_config = os.path.join(pretraining_adapter_path, "adapter_config.json")
    dst_config = "./adapters/phase1/adapter_config.json"
    if os.path.exists(src_config):
        shutil.copy2(src_config, dst_config)
    
    print("✅ รอบที่ 1 สำเร็จ!")

    # ================================================================
    # โหลดโมเดลใหม่ + Adapter จากรอบที่ 1 เพื่อเริ่มรอบที่ 2
    # ================================================================
    print("\n🔄 กำลังโหลด Adapter จากรอบที่ 1 สำหรับรอบที่ 2...")
    model, tokenizer = load(model_id)
    model.freeze()
    model = load_adapters(model, "./adapters/phase1")
    
    # ================================================================
    # รอบที่ 2: เรียนรู้ข้อมูลบุญส่งเป็นภาษาอีสานจาก data/raw/ (ข้อมูลเล็ก)
    # ================================================================
    run_training_phase(
        model=model,
        tokenizer=tokenizer,
        data_dir="./data/raw",
        adapter_file="./adapters/adapters.safetensors",
        phase_name="รอบที่ 2: เรียนรู้ข้อมูลบุญส่ง (ภาษาอีสาน)",
        iters=60
    )
    print("✅ รอบที่ 2 สำเร็จ!")

    # ================================================================
    # สร้าง adapter_config.json ใน ./adapters/ (สำเนาจาก Pretraining)
    # ================================================================
    adapter_dir = "./adapters"
    os.makedirs(adapter_dir, exist_ok=True)
    
    src_config = os.path.join(pretraining_adapter_path, "adapter_config.json")
    dst_config = os.path.join(adapter_dir, "adapter_config.json")
    
    if os.path.exists(src_config) and not os.path.exists(dst_config):
        shutil.copy2(src_config, dst_config)
        print(f"📋 สร้าง adapter_config.json ใน {adapter_dir}/ เรียบร้อย")
    
    print("\n🎉 เสร็จสิ้นทั้ง 2 รอบ! Adapter สุดท้ายอยู่ที่ ./adapters/adapters.safetensors")


if __name__ == "__main__":
    main()