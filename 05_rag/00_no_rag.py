"""
สคริปต์นี้แสดงการถามคำถามโดยไม่ใช้ RAG (Retrieval-Augmented Generation)

📌 ค่า config ที่สำคัญ:
  • temp=0.6 → ความ “สร้างสรรค์” ของคำตอบ
    - ถ้าค่าน้อย (0.1): คำตอบแน่นอน แต่ซ้ำซาก
    - ถ้าค่ามาก (1.0): คำตอบสร้างสรรค์ แต่อาจเพี้ยน
    - 0.6 เป็นค่ากลางๆ ที่สมดุล

  • top_p=0.9 → จำกัดเฉพาะ Token ที่มีโอกาสสะสม 90%
    - ถ้าค่าน้อย: คำตอบแคบ แต่แน่นอน
    - ถ้าค่ามาก: คำตอบกว้าง แต่อาจเพี้ยน
    - 0.9 เป็นค่ามาตรฐาน

💡 ผลต่อ Inference:
  - ไม่ใช้ RAG → โมเดลตอบจากความรู้ที่มีอยู่แล้ว
  - ถ้าคำถามเกี่ยวกับข้อมูลเฉพาะ → คำตอบอาจผิด
  - ถ้าคำถามเกี่ยวกับความรู้ทั่วไป → คำตอบอาจถูกต้อง
"""
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")
sampler = make_sampler(temp=0.6, top_p=0.9)


def ask(query):
    """ถามคำถามเกี่ยวกับ TOR TH-AI Passport โดยไม่มี RAG"""

    prompt = f"""คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับโครงการ TH-AI Passport
หากไม่ทราบ บอกว่าไม่ทราบ

คำถาม: {query}
คำตอบ:"""

    return generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=130)


if __name__ == "__main__":
    q = "วัตถุประสงค์ของโครงการ TH-AI Passport"
    print(f"คำถาม: {q}")
    print(f"คำตอบ: {ask(q)}")
