"""
Majority Voting over Chain-of-Thought Example
ตัวอย่างการใช้งาน Majority Voting over CoT
"""

from mlx_lm import load, generate
from collections import Counter

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "../01_pretraining/adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)


# ============================================================
# Majority Voting over CoT คืออะไร?
# ============================================================
# 1. รัน Chain-of-Thought หลายๆ ครั้ง (เช่น 5 ครั้ง)
# 2. เก็บคำตอบจากแต่ละรอบ
# 3. เลือกคำตอบที่ได้บ่อยที่สุด (Majority Vote)
# ============================================================


# Prompt แบบ Chain-of-Thought
cot_prompt = """
คำถาม: บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?

ให้คิดเป็นขั้นตอนทีละขั้น:

ขั้นที่ 1: ระบุข้อมูลบุคคล
- บุญส่ง ศรีทอง เป็นใคร?
- มีข้อมูลพื้นฐานอะไรบ้างเกี่ยวกับเขา?

ขั้นที่ 2: การศึกษา
- เขาเรียนจบระดับไหน?
- เรียนสาขาอะไร?
- จบปีไหน?

ขั้นที่ 3: ค้นหางานแรก
- หลังเรียนจบ เขาเริ่มทำงานปีอะไร?
- ทำงานที่ไหน?

ขั้นที่ 4: สรุปคำตอบ
- ระบุปีที่เริ่มทำงานครั้งแรก

คำตอบ:
"""


# ============================================================
# รัน CoT หลายครั้ง (Majority Voting)
# ============================================================

num_votes = 5  # จำนวนรอบที่จะโหวต
answers = []

print("=" * 60)
print(f"Majority Voting over CoT ({num_votes} rounds)")
print("=" * 60)

for i in range(num_votes):
    print(f"\nรอบที่ {i + 1}/{num_votes}")
    print("-" * 60)
    
    response = generate(
        model,
        tokenizer,
        prompt=cot_prompt,
        max_tokens=128,
        verbose=False
    )
    
    # ดึงคำตอบ (อาจต้อง parse ตามรูปแบบ)
    answer = response.strip()
    answers.append(answer)
    
    print(f"คำตอบ: {answer}")


# ============================================================
# โหวตหาคำตอบที่ได้บ่อยที่สุด
# ============================================================

print("\n" + "=" * 60)
print("ผลการโหวต (Majority Voting)")
print("=" * 60)

# นับความถี่ของแต่ละคำตอบ
# (ในระบบจริง ควร extract เฉพาะส่วนที่เป็นคำตอบ เช่น ปี พ.ศ.)
answer_counts = Counter(answers)

print("\nคำตอบทั้งหมด:")
for answer, count in answer_counts.most_common():
    print(f"  - ได้ {count} คะแนน: {answer[:100]}...")

# เลือกคำตอบที่ได้คะแนนสูงสุด
if answer_counts:
    best_answer = answer_counts.most_common(1)[0][0]
    print("\n" + "=" * 60)
    print(f"คำตอบสุดท้าย (Majority Vote): {best_answer}")
    print("=" * 60)


# ============================================================
# ตัวอย่างการ parse คำตอบ (ถ้าต้องการเฉพาะปี)
# ============================================================

print("\n" + "=" * 60)
print("ตัวอย่าง: การ extract ปีจากคำตอบ")
print("=" * 60)

import re

def extract_year(text):
    """extract ปี พ.ศ. จากข้อความ"""
    # ค้นหารูปแบบ ปี พ.ศ. XXXX หรือ ตัวเลข 4 หลัก
    patterns = [
        r'ปี\s*พ\.?ศ\.?\s*(\d{4})',
        r'(\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None


# ตัวอย่างการใช้งาน
print("\nคำตอบที่ถูก extract:")
for i, answer in enumerate(answers, 1):
    year = extract_year(answer)
    if year:
        print(f"  รอบที่ {i}: ปี {year}")
    else:
        print(f"  รอบที่ {i}: ไม่พบปี")


# โหวตปีที่ได้บ่อยที่สุด
years = [extract_year(ans) for ans in answers]
years = [y for y in years if y is not None]

if years:
    year_counts = Counter(years)
    best_year = year_counts.most_common(1)[0][0]
    print(f"\nปีที่ได้คะแนนสูงสุด: พ.ศ. {best_year}")
    print(f"คะแนน: {year_counts.most_common(1)[0][1]}/{num_votes}")

# สรุป Token Usage
total_prompt_tokens = len(tokenizer.encode(cot_prompt)) * num_votes
total_response_tokens = sum(len(tokenizer.encode(ans)) for ans in answers)
print(f"\n{'='*40}")
print(f"Token Usage Summary")
print(f"{'='*40}")
print(f"Prompt tokens:   {total_prompt_tokens} ({num_votes} calls)")
print(f"Response tokens: {total_response_tokens}")
print(f"Total tokens:    {total_prompt_tokens + total_response_tokens}")
