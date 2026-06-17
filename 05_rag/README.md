# Module 5 — RAG (Retrieval-Augmented Generation)

ให้โมเดลตอบคำถามจากเอกสาร PDF โดยใช้ ChromaDB เป็น Vector Store

---

## ไฟล์ในโมดูล

| ไฟล์ | คำอธิบาย |
|---|---|
| `00_no_rag.py` | ถามโมเดลโดยตรง (ไม่มี RAG) เพื่อเปรียบเทียบ |
| `01_parse_data.py` | อ่านข้อความจาก PDF |
| `02_local_vector_store.py` | สร้าง ChromaDB และ ingest ข้อมูล |
| `03_inference.py` | RAG inference — ค้นหา context แล้วตอบคำถาม |

---

## วิธีรัน

```bash
uv run python 00_no_rag.py
uv run python 01_parse_data.py
uv run python 02_local_vector_store.py
uv run python 03_inference.py
```

---

## โครงสร้างข้อมูล

```
05_rag/
├── pdf/
│   └── TOR-TH-AI-Pasport.pdf
└── tor_db/
    └── chroma.sqlite3
```
