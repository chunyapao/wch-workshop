from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Load the model and tokenizer
model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")

sampler = make_sampler(temp=0.6, top_p=0.9)


def ask(query):
    """ถามคำถามเกี่ยวกับ TOR TH-AI Passport โดยไม่มี RAG"""

    prompt = f"""คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับโครงการ TH-AI Passport
หากไม่ทราบ บอกว่าไม่ทราบ

คำถาม: {query}
คำตอบ:"""

    return generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=500)


if __name__ == "__main__":
    q = "วัตถุประสงค์ของโครงการ TH-AI Passport"
    print(f"คำถาม: {q}")
    print(f"คำตอบ: {ask(q)}")
