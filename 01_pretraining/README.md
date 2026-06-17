# Module 1 — Pretraining

สอนการ pretrain LLM บนข้อมูลที่กำหนดเองโดยใช้ MLX และ LoRA

---

## ไฟล์ในโมดูล

| ไฟล์ | คำอธิบาย |
|---|---|
| `00_base-model.py` | ทดสอบโมเดลพื้นฐาน (ก่อน pretrain) |
| `01_data-preparation.py` | เตรียมข้อมูลดิบจาก dataset/ |
| `02_packaging-data.py` | แพ็คข้อมูลเป็น block ขนาด 128 tokens |
| `03_model-initialization.py` | ตรวจสอบโมเดลพื้นฐาน |
| `04_training-run.py` | รัน Pretraining (LoRA) |
| `05_adapter.py` | ทดสอบโมเดลหลัง pretrain |

---

## วิธีรัน

```bash
uv run python 00_base-model.py
uv run python 01_data-preparation.py
uv run python 02_packaging-data.py
uv run python 04_training-run.py
uv run python 05_adapter.py
```

---

## โครงสร้างข้อมูล

```
01_pretraining/
├── dataset/
│   ├── ex-data-quality.txt
│   └── profile-th.txt
├── data/
│   ├── raw/
│   │   ├── train.jsonl
│   │   └── valid.jsonl
│   └── packed/
│       ├── train.jsonl
│       └── valid.jsonl
└── adapters/
    ├── adapter_config.json
    └── adapters.safetensors
```
