# Module 2 — Fine-tuning

Fine-tune โมเดลด้วย Instruction Tuning บนข้อมูลภาษาอีสาน โดยต่อยอดจาก adapter ที่ pretrain ไว้

รองรับ 2 ขั้นตอนการเทรน:
1. **รอบที่ 1**: เรียนรู้ภาษาอีสานจาก `data/dataset/` (ข้อมูลใหญ่)
2. **รอบที่ 2**: เรียนรู้ข้อมูลบุญส่งเป็นภาษาอีสานจาก `data/raw/` (ข้อมูลเล็ก)

---

## ไฟล์ในโมดูล

| ไฟล์ | คำอธิบาย |
|---|---|
| `data-preparation.py` | ดาวน์โหลดและเตรียม dataset ภาษาอีสานจาก HuggingFace |
| `00_convert-data.py` | แปลงข้อมูลดิบเป็นรูปแบบ prompt/completion (ใช้ regex แปลงไทย→อีสาน) |
| `01_training-run.py` | รัน Fine-tuning (LoRA) รอบเดียว |
| `01_training-run-isan.py` | รัน Fine-tuning 2 รอบ (ภาษาอีสาน + ข้อมูลบุญส่ง) |
| `00_test-isan.py` | ทดสอบ Tree of Thoughts ภาษาอีสาน |
| `03_instruction-tuning.py` | ทดสอบ Instruction Tuning ด้วย adapter ที่ fine-tune แล้ว |

---

## วิธีรัน

```bash
# เตรียมข้อมูล
uv run python data-preparation.py   # ดาวน์โหลด dataset ภาษาอีสาน
uv run python 00_convert-data.py    # แปลงข้อมูลเป็นภาษาอีสาน

# เทรนรอบเดียว
uv run python 01_training-run.py

# หรือ เทรน 2 รอบ (แนะนำ)
uv run python 01_training-run-isan.py

# ทดสอบหลัง fine-tune
uv run python 00_test-isan.py
uv run python 03_instruction-tuning.py
```

---

## โครงสร้างข้อมูล

```
02_finetuning/
├── data/
│   ├── raw/              ← ข้อมูลบุญส่ง (ภาษาอีสาน)
│   │   ├── train.jsonl
│   │   └── valid.jsonl
│   └── dataset/          ← ข้อมูลภาษาอีสานจาก HuggingFace
│       ├── train.jsonl
│       └── valid.jsonl
└── adapters/
    ├── adapter_config.json
    └── adapters.safetensors
```
