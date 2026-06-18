"""
สคริปต์นี้รันการ Fine-tuning โมเดลด้วย Adapter จาก Pretraining

📌 ค่า config ที่สำคัญ:
  • adapter_path = "../01_pretraining/adapters" → โฟลเดอร์ที่เก็บ Adapter จาก Pretraining
    - ต้องมีไฟล์ adapters.safetensors และ adapter_config.json
    - ถ้าไม่มี → โหลดไม่ได้ หรือคำตอบไม่เปลี่ยนจาก Base Model

  • max_seq_length=512 → ความยาวสูงสุดของข้อความที่โมเดลเห็นตอนเทรน
    - ถ้าค่ามาก: โมเดลเห็นบริบทยาว แต่ใช้ RAM มาก
    - ค่าน้อย: โมเดลเห็นบริบทสั้น แต่ใช้ RAM น้อย
    - 512 เหมาะกับข้อความยาวๆ เช่น บทสนทนา

  • learning_rate=1e-4 → อัตราการเรียนรู้
    - ถ้าค่ามาก: เรียนรู้เร็ว แต่อาจ “ข้าม” จุดที่ดีที่สุด
    - ค่าน้อย: เรียนรู้ช้า แต่แม่นยำกว่า
    - 1e-4 (0.0001) เป็นค่ามาตรฐานสำหรับ Fine-tuning

  • prompt_feature="prompt", completion_feature="completion"
    - บอกโมเดลว่า “คำถาม” อยู่ที่ไหน และ “คำตอบ” อยู่ที่ไหน
    - โมเดลจะเรียนรู้เฉพาะส่วน completion (ไม่เรียนรู้ส่วน prompt)
    - สำคัญมากสำหรับ Instruction Tuning

💡 ผลต่อ Inference:
  - Adapter ที่เทรนแล้ว → คำตอบเปลี่ยนไป (มีความรู้ใหม่)
  - max_seq_length → ความยาวบริบทที่โมเดล “จำ” ได้ตอน inference
  - prompt/completion format → โมเดลเรียนรู้รูปแบบเฉพาะ ถ้าตอน inference ใช้รูปแบบเดียวกัน → คำตอบแม่นยำ
"""
import os
import types
import mlx.optimizers as optim
import mlx.utils
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, train
from mlx_lm.tuner.utils import load_adapters
from mlx_lm.tuner.datasets import load_dataset, CacheDataset

def main():
    # 1. ระบุชื่อโมเดล 4-bit และตำแหน่งข้อมูล
    model_id = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
    adapter_path = "../01_pretraining/adapters"  # ใช้ adapter จาก pretraining ที่เทรนไว้
    data_dir = "./data/raw" # ชี้ไปที่โฟลเดอร์ข้อมูลดิบที่เราเตรียมไว้
    
    # 2. โหลดโมเดลพื้นฐาน (ไม่มี adapter) แล้วค่อยโหลด adapter จาก pretraining
    model, tokenizer = load(model_id)
    model.freeze()
    
    # load_adapters จะ: สร้าง LoRA layers ใหม่ (เฉพาะ lora_a/lora_b เทรนได้) + โหลดน้ำหนักจาก pretraining
    model = load_adapters(model, adapter_path)
    
    trainable_params = mlx.utils.tree_flatten(model.trainable_parameters())
    print(f"✅ มีพารามิเตอร์สำหรับเทรน: {len(trainable_params)} รายการ")
    
    # 3. โหลดชุดข้อมูลสำหรับ Instruction Tuning
    # การระบุ prompt_feature และ completion_feature จะทำให้ MLX-LM ทำ Masking Loss ให้คำถามอัตโนมัติ
    args = types.SimpleNamespace(
        data=data_dir,
        train=True,
        test=False,
        prompt_feature="prompt",         # ให้โมเดลอ่านคำถามจาก Key ชื่อ "prompt"
        completion_feature="completion"  # ให้โมเดลเรียนรู้คำตอบจาก Key ชื่อ "completion"
    )
    
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    
    # 4. ตั้งค่า Hyperparameters สำหรับเทรนบน M1
    training_args = TrainingArgs(
        adapter_file="./adapters/adapters.safetensors",
        iters=100,              # จำนวนรอบในการเทรน (หากมีเวลาสามารถปรับเป็น 1000-3000 ได้)
        batch_size=1,            # ล็อคไว้ที่ 1 เพื่อป้องกัน Out of Memory บน RAM 8GB
        max_seq_length=512,      
        steps_per_eval=10,      # แวะประเมินผล Valid Loss ทุกๆ 10 steps
        steps_per_save=50,      # บันทึก Checkpoint 
    )
    
    optimizer = optim.Adam(learning_rate=1e-4)
    
    # 5. คลาสสำหรับรายงานผลการเทรน (โชว์ Loss สวยๆ บน Terminal)
    class MetricsReporter:
        def on_train_loss_report(self, info):
            print(f"🔄 Step {info['iteration']:04d} | Train Loss: {info['train_loss']:.4f} | ความเร็ว: {info['iterations_per_second']:.2f} it/sec")
            
        def on_val_loss_report(self, info):
            print(f"\n✨ ---> Validation Loss: {info['val_loss']:.4f} <---✨\n")
            
    metrics = MetricsReporter()
    
    # 6. เริ่มลุยเทรน!
    print("\n🚀 เริ่มกระบวนการเทรนให้น้องเว้าอีสาน...")
    os.makedirs("adapters", exist_ok=True)
    
    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=metrics
    )
    

if __name__ == "__main__":
    main()