from mlx_lm import load, generate

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "../01_pretraining/adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = """
คำถาม: บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?

ให้คิดเป็นขั้นตอนทีละขั้น:

ขั้นที่ 1: ระบุข้อมูลบุคคล
- บุญส่ง ศรีทอง เป็นใคร?
- เกิดที่ไหน? เมื่อไร?
- มีข้อมูลพื้นฐานอะไรบ้างเกี่ยวกับเขา?

ขั้นที่ 2: การศึกษา
- เขาเรียนจบระดับไหน?
- เรียนสาขาอะไร?
- จบจากสถาบันใด?
- จบปีไหน?

ขั้นที่ 3: ทักษะและความสามารถ
- เขามีทักษะด้านอะไรบ้าง?
- ความเชี่ยวชาญพิเศษคืออะไร?
- มีประสบการณ์อะไรก่อนทำงาน?

ขั้นที่ 4: ค้นหางานแรก
- หลังเรียนจบ เขาเริ่มหางานเมื่อไร?
- สมัครงานประเภทไหน?
- มีบริษัทใดบ้างที่เขาสัมภาษณ์?

ขั้นที่ 5: ยืนยันสถานที่ทำงานแรก
- เขาเริ่มทำงานที่ไหนเป็นที่แรก?
- ตำแหน่งอะไร?
- ทำหน้าที่อะไรบ้าง?
- ทำงานที่นั่นนานแค่ไหน?
- ได้เรียนรู้อะไรจากงานแรก?

ขั้นที่ 6: ตรวจสอบความถูกต้อง
- ข้อมูลที่ได้มาถูกต้องหรือไม่?
- มีแหล่งใดยืนยันได้?
- มีข้อมูลอื่นเพิ่มเติมหรือไม่?

คำตอบ:
"""


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
