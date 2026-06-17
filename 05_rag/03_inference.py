import importlib.util

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Import 02_local_vector_store.py
_spec = importlib.util.spec_from_file_location("vector_store", "02_local_vector_store.py")
_vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vs)
query_vs = _vs.query

# Load the model and tokenizer
model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")

sampler = make_sampler(temp=0.6, top_p=0.9)


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
