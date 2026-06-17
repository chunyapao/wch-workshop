# Module 3 — Prompt Engineering

เทคนิคต่างๆ ในการเขียน Prompt เพื่อดึงประสิทธิภาพของโมเดลออกมา

---

## ไฟล์ในโมดูล

| ไฟล์ | เทคนิค | คำอธิบาย |
|---|---|---|
| `01_zero-shot.py` | Zero-Shot | ถามตรงโดยไม่มีตัวอย่าง |
| `02_few-shot.py` | Few-Shot | ให้ตัวอย่าง 2-3 ข้อก่อนถาม |
| `03_chain-of-thought.py` | Chain-of-Thought | ให้โมเดลคิดแบบ step-by-step |
| `04_majority-voting-cot.py` | Majority Voting CoT | รันหลายรอบแล้วเลือกคำตอบที่ชนะ |
| `05_tree-of-thoughts.py` | Tree of Thoughts | สร้างหลายเส้นทางความคิดแล้วประเมิน |
| `06_tree-of-thoughts-isan.py` | ToT (ภาษาอีสาน) | Tree of Thoughts ด้วยภาษาอีสาน |

> ทุกสคริปต์แสดงสรุป Token Usage (prompt / response / total) หลังรันเสร็จ

---

## วิธีรัน

```bash
uv run python 01_zero-shot.py
uv run python 02_few-shot.py
uv run python 03_chain-of-thought.py
uv run python 04_majority-voting-cot.py
uv run python 05_tree-of-thoughts.py
uv run python 06_tree-of-thoughts-isan.py
```
