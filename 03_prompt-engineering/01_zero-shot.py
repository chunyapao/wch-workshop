from mlx_lm import load, generate

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "../01_pretraining/adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = "บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?"

# หรือรูปแบบที่มีโครงสร้างชัดเจน
prompt = """
คำถาม: บุญส่ง ศรีทอง ทำงานที่ odds  ทำงานที่แรกที่ไหน?
คำตอบ:
"""

# ฟลักการทำ Zero-Shot Prompting ถามคำถามโดยตรง ไม่ต้องมีตัวอย่างประกอบ ให้โมเดลตอบจากความรู้ที่มีอยู่แล้ว

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=128,
    verbose=True
)

# สรุป Token Usage
prompt_tokens = len(tokenizer.encode(prompt))
response_tokens = len(tokenizer.encode(response))
print(f"\n{'='*40}")
print(f"Token Usage Summary")
print(f"{'='*40}")
print(f"Prompt tokens:   {prompt_tokens}")
print(f"Response tokens: {response_tokens}")
print(f"Total tokens:    {prompt_tokens + response_tokens}")
