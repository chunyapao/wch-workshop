"""
สคริปต์นี้แสดงการถามคำถามโดยใช้ RAG (Retrieval-Augmented Generation)

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
  - ใช้ RAG → โมเดลตอบจากข้อมูลใน Vector Store + ความรู้ที่มีอยู่แล้ว
  - ถ้าข้อมูลใน Vector Store มีคุณภาพ → คำตอบแม่นยำ
  - ถ้าข้อมูลใน Vector Store ไม่มีคุณภาพ → คำตอบผิดเพี้ยน
  - RAG ช่วยลดโอกาส “น้ำท่วม” (hallucinate) เพราะมีข้อมูลอ้างอิง
"""


def ask(query):
    """ถามคำถามเกี่ยวกับ TOR TH-AI Passport โดยใช้ RAG"""

    # 1. ค้นหา context จาก ChromaDB
    results = query_vs(query)
    context = results['documents'][0][0]

    # 2. สร้าง prompt
    prompt = f"""คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับโครงการ TH-AI Passport
ใช้ข้อมูลด้านล่างเท่านั้นในการตอบ หากไม่มีข้อมูล บอกว่าไม่ทราบ

ข้อมูลอ้างอิง:
{context}

คำถาม: {query}
คำตอบ:"""

    # 3. สร้างคำตอบด้วย MLX
    return generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=500)


if __name__ == "__main__":
    q = "วัตถุประสงค์ของโครงการ TH-AI Passport"
    print(f"คำถาม: {q}")
    print(f"คำตอบ: {ask(q)}")
