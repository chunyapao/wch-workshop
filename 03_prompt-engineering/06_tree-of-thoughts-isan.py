"""
สคริปต์นี้แสดงการใช้งาน Tree of Thoughts (ToT) ภาษาอีสาน

📌 ค่า config ที่สำคัญ:
  • max_tokens=256 (แต่ละเส้นทาง) → จำกัดคำตอบแต่ละเส้นทาง 256 Token
  • max_tokens=512 (ประเมิน) → จำกัดคำตอบตอนประเมิน 512 Token
    - ถ้าค่าน้อย: คำตอบสั้น อาจไม่ครบถ้วน
    - ถ้าค่ามาก: คำตอบยาว แต่ใช้เวลานาน

💡 ผลต่อ Inference:
  - ToT = สร้างหลายเส้นทางความคิด แล้วเลือกเส้นทางที่ดีที่สุด
  - แต่ละเส้นทางคิดแบบต่างกัน (การศึกษา, ประวัติ, บริบท)
  - โมเดลจะประเมินแต่ละเส้นทาง แล้วเลือกเส้นทางที่น่าเชื่อถือที่สุด
  - เหมาะกับคำถามที่ซับซ้อน ต้องใช้การวิเคราะห์หลายมุมมอง
  - max_tokens ที่มากพอ → คำตอบมีเหตุผลครบถ้วน
"""

from mlx_lm import load, generate

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "../01_pretraining/adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)


# ============================================================
# Tree of Thoughts คืออี่หยัง?
# ============================================================
# 1. สร้างหลายเส้นทางความคิด (branches)
# 2. แต่ละเส้นทางคิดแบบต่างกัน
# 3. ประเมินแต่ละเส้นทาง
# 4. เลือกเส้นทางที่ดีที่สุด
# ============================================================


# ============================================================
# ขั้นตอนที่ 1: สร้างเส้นทางความคิดหลายเส้นทาง
# ============================================================

print("=" * 60)
print("Tree of Thoughts - บุญส่ง ศรีทอง เฮ็ดงานแรกปีได?")
print("=" * 60)


# เส้นทางที่ 1: คิดจากข้อมูลการศึกษา
branch_1_prompt = """
คำถาม: บุญส่ง ศรีทอง เฮ็ดงานแรกปีได?

เส้นทางที่ 1: วิเคราะห์จากข้อมูลการศึกษา
- บุญส่ง ศรีทอง เรียนจบปีได?
- หลังเรียนจบ เริ่มเฮ็ดงานเลยบ่?
- ถ้าเริ่มเฮ็ดงานเลย ปีที่เริ่มเฮ็ดงานคือปีได?

คิดวิเคราะห์:
"""

print("\n" + "=" * 60)
print("เส้นทางที่ 1: วิเคราะห์จากข้อมูลการศึกษา")
print("=" * 60)

response_1 = generate(
    model,
    tokenizer,
    prompt=branch_1_prompt,
    max_tokens=256,
    verbose=False
)

branch_1_answer = response_1.strip()
print(branch_1_answer)


# เส้นทางที่ 2: คิดจากข้อมูลประวัติการทำงาน
branch_2_prompt = """
คำถาม: บุญส่ง ศรีทอง เฮ็ดงานแรกปีได?

เส้นทางที่ 2: วิเคราะห์จากประวัติการเฮ็ดงาน
- บุญส่ง ศรีทอง มีประวัติการเฮ็ดงานจั่งใด?
- งานแรกคืองานอี่หยัง?
- เริ่มเฮ็ดงานแรกปีได?

คิดวิเคราะห์:
"""

print("\n" + "=" * 60)
print("เส้นทางที่ 2: วิเคราะห์จากประวัติการเฮ็ดงาน")
print("=" * 60)

response_2 = generate(
    model,
    tokenizer,
    prompt=branch_2_prompt,
    max_tokens=256,
    verbose=False
)

branch_2_answer = response_2.strip()
print(branch_2_answer)


# เส้นทางที่ 3: คิดจากข้อมูลบริบทแวดล้อม
branch_3_prompt = """
คำถาม: บุญส่ง ศรีทอง เฮ็ดงานแรกปีได?

เส้นทางที่ 3: วิเคราะห์จากบริบทแวดล้อม
- บุญส่ง ศรีทอง เกิดปีได?
- อายุเถ่าได๋เมื่อเริ่มเฮ็ดงาน?
- ช่วงเวลานั้นมีเหตุการณ์อี่หยังที่เกี่ยวข้อง?
- สรุปปีที่เริ่มเฮ็ดงานคือปีได?

คิดวิเคราะห์:
"""

print("\n" + "=" * 60)
print("เส้นทางที่ 3: วิเคราะห์จากบริบทแวดล้อม")
print("=" * 60)

response_3 = generate(
    model,
    tokenizer,
    prompt=branch_3_prompt,
    max_tokens=256,
    verbose=False
)

branch_3_answer = response_3.strip()
print(branch_3_answer)


# ============================================================
# ขั้นตอนที่ 2: ประเมินแต่ละเส้นทาง
# ============================================================

print("\n" + "=" * 60)
print("ขั้นตอนการประเมิน (Evaluation)")
print("=" * 60)

evaluation_prompt = f"""
ฉันมี 3 เส้นทางความคิดเกี่ยวกับคำถาม: "บุญส่ง ศรีทอง เฮ็ดงานแรกปีได?"

เส้นทางที่ 1 (วิเคราะห์จากข้อมูลการศึกษา):
{branch_1_answer}

เส้นทางที่ 2 (วิเคราะห์จากประวัติการเฮ็ดงาน):
{branch_2_answer}

เส้นทางที่ 3 (วิเคราะห์จากบริบทแวดล้อม):
{branch_3_answer}

จงประเมินว่า:
1. เส้นทางไหนน่าเชื่อถือที่สุด?
2. ทำไมถึงเลือกเส้นทางนั้น?
3. คำตอบสุดท้ายที่ได้จากเส้นทางนั้นคือปีได?

การประเมิน:
"""

print("\nกำลังประเมินเส้นทางทั้งหมด...")
print("-" * 60)

evaluation_response = generate(
    model,
    tokenizer,
    prompt=evaluation_prompt,
    max_tokens=512,
    verbose=False
)

print(evaluation_response.strip())


# ============================================================
# ขั้นตอนที่ 3: สรุปคำตอบสุดท้าย
# ============================================================

print("\n" + "=" * 60)
print("คำตอบสุดท้าย (Final Answer)")
print("=" * 60)

final_prompt = """
จากการวิเคราะห์ 3 เส้นทางเกี่ยวกับคำถาม: "บุญส่ง ศรีทอง เฮ็ดงานแรกปีได?"

จงสรุปคำตอบสุดท้าย:
- ระบุปีที่เริ่มเฮ็ดงานครั้งแรก
- ให้เหตุผลสั้นๆ
- ตอบให้ชัดเจน

คำตอบสุดท้าย:
"""

final_response = generate(
    model,
    tokenizer,
    prompt=final_prompt,
    max_tokens=128,
    verbose=False
)

print(final_response.strip())


# ============================================================
# สรุปโครงสร้าง Tree of Thoughts
# ============================================================

print("\n" + "=" * 60)
print("สรุปโครงสร้าง Tree of Thoughts")
print("=" * 60)

print("""
Tree of Thoughts Structure:
                           [คำถาม]
                              |
              ┌───────────────┼───────────────┐
              |               |               |
         [เส้นทางที่ 1]  [เส้นทางที่ 2]  [เส้นทางที่ 3]
         การศึกษา       ประวัติการเฮ็ดงาน   บริบทแวดล้อม
              |               |               |
              └───────────────┼───────────────┘
                              |
                        [ประเมิน]
                              |
                        [คำตอบสุดท้าย]

ประโยชน์:
- คิดได้หลายมุมมอง
- เปรียบเทียบความน่าเชื่อถือ
- ได้คำตอบที่ดีที่สุด
- เหมาะกับปัญหาที่ซับซ้อน
""")

# สรุป Token Usage
p1_tokens = len(tokenizer.encode(branch_1_prompt))
p2_tokens = len(tokenizer.encode(branch_2_prompt))
p3_tokens = len(tokenizer.encode(branch_3_prompt))
eval_tokens = len(tokenizer.encode(evaluation_prompt))
final_tokens = len(tokenizer.encode(final_prompt))
total_prompt_tokens = p1_tokens + p2_tokens + p3_tokens + eval_tokens + final_tokens

r1_tokens = len(tokenizer.encode(response_1))
r2_tokens = len(tokenizer.encode(response_2))
r3_tokens = len(tokenizer.encode(response_3))
eval_r_tokens = len(tokenizer.encode(evaluation_response))
final_r_tokens = len(tokenizer.encode(final_response))
total_response_tokens = r1_tokens + r2_tokens + r3_tokens + eval_r_tokens + final_r_tokens

print(f"{'='*40}")
print(f"Token Usage Summary")
print(f"{'='*40}")
print(f"Prompt tokens:   {total_prompt_tokens} (5 calls)")
print(f"Response tokens: {total_response_tokens}")
print(f"Total tokens:    {total_prompt_tokens + total_response_tokens}")
