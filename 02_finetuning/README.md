# Module 2 — Fine-tuning

Fine-tune โมเดลด้วย Instruction Tuning บนข้อมูลภาษาอีสาน โดยต่อยอดจาก adapter ที่ pretrain ไว้

---

## ไฟล์ในโมดูล

| ไฟล์ | คำอธิบาย |
|---|---|
| `data-preparation.py` | ดาวน์โหลดและเตรียม dataset ภาษาอีสานจาก HuggingFace |
| `00_convert-data.py` | แปลงข้อมูลดิบเป็นรูปแบบ prompt/completion |
| `01_training-run.py` | รัน Fine-tuning (LoRA) |
| `02_tree-of-thoughts-isan.py` | ทดสอบ Tree of Thoughts ด้วยภาษาอีสาน |
| `03_instruction-tuning.py` | ทดสอบ Instruction Tuning ด้วย adapter ที่ fine-tune แล้ว |

---

## วิธีรัน

```bash
uv run python data-preparation.py   # ดาวน์โหลด dataset ภาษาอีสาน
uv run python 00_convert-data.py
uv run python 01_training-run.py
uv run python 03_instruction-tuning.py   # ทดสอบหลัง fine-tune
```

---

## โครงสร้างข้อมูล

```
02_finetuning/
├── data/
│   ├── raw/
│   │   ├── train.jsonl
│   │   └── valid.jsonl
│   └── dataset/
│       ├── train.jsonl
│       └── valid.jsonl
└── adapters/
    ├── adapter_config.json
    └── adapters.safetensors
```
