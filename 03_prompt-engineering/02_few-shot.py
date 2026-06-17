from mlx_lm import load, generate

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "../01_pretraining/adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = """
ตัวอย่างที่ 1:
คำถาม: สมชาย ใจดี ทำงานที่แรกที่ไหน?
คำตอบ: สมชาย ใจดี เริ่มทำงานครั้งแรกที่บริษัท ABC จำกัด ในตำแหน่งวิศวกรซอฟต์แวร์ เมื่อปี 2550

ตัวอย่างที่ 2:
คำถาม: วิภา รักเรียน ทำงานที่แรกที่ไหน?
คำตอบ: วิภา รักเรียน เริ่มทำงานครั้งแรกที่โรงพยาบาลเชียงใหม่ ในตำแหน่งพยาบาล เมื่อปี 2555

ตัวอย่างที่ 3:
คำถาม: ประเสริฐ มั่นคง ทำงานที่แรกที่ไหน?
คำตอบ: ประเสริฐ มั่นคง เริ่มทำงานแรกที่ธนาคารกรุงไทย สาขากรุงเทพฯ ในตำแหน่งพนักงานบริการลูกค้า เมื่อปี 2548

คำถาม: บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?
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
