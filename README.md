# WCH Workshop — Local LLM & AI Engineering

Workshop เรียนรู้การทำงานกับ LLM แบบ Local บน Apple Silicon (MLX) ครอบคลุมตั้งแต่ Pretraining, Fine-tuning, Prompt Engineering, Local Server, RAG ไปจนถึง MCP

---

## Requirements

| รายการ | รายละเอียด |
|---|---|
| เครื่อง | Apple Silicon (M1/M2/M3/M4/M5) |
| Python | 3.11+ |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| RAM | 8 GB ขึ้นไป (แนะนำ 16 GB) |

---

## เริ่มต้น (ต้องมีอินเทอร์เน็ตครั้งแรก)

### 1. ติดตั้ง uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> หลังติดตั้งเสร็จให้เปิด terminal ใหม่ หรือรัน `source ~/.bashrc` / `source ~/.zshrc` เพื่อให้ใช้คำสั่ง `uv` ได้

### 2. ติดตั้ง Python

```bash
uv python install 3.11
```

### 3. ตั้งค่า Python version ของ project

```bash
uv python pin 3.11
```

> คำสั่งนี้จะสร้างไฟล์ `.python-version` ในโปรเจกต์ เพื่อให้ `uv` ใช้ Python เวอร์ชันที่กำหนดโดยอัตโนมัติ

### 4. ติดตั้ง dependencies + ดาวน์โหลด models

```bash
uv run python setup.py
```

สคริปต์จะทำ 4 ขั้นตอนโดยอัตโนมัติ:

1. `uv sync` — ติดตั้ง Python packages ทั้งหมด
2. ดาวน์โหลด LLM หลัก → `typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit`
3. ดาวน์โหลด Embedding model → `all-MiniLM-L6-v2`
4. ตรวจสอบ imports ทุก package

หลังจากนั้นสามารถทำ workshop ได้แบบ **offline** (ยกเว้น Module 6 ที่ดึงข้อมูลหุ้นแบบ realtime)

---

## โครงสร้าง Workshop

```
wch-workshop/
├── setup.py                  ← รันก่อน (เตรียม environment)
├── 01_pretraining/           ← Module 1: Pretraining
├── 02_finetuning/            ← Module 2: Fine-tuning (LoRA)
├── 03_prompt-engineering/    ← Module 3: Prompt Engineering
├── 04_local-llms/            ← Module 4: Local LLM Server
├── 05_rag/                   ← Module 5: RAG
└── 06_mcp/                   ← Module 6: MCP (Model Context Protocol)
```
---

## Models ที่ใช้

| โมเดล | ใช้ใน | Cache location |
|---|---|---|
| `typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit` | ทุก module | `~/.cache/huggingface/hub/` |
| `all-MiniLM-L6-v2` | Module 5 (RAG) | `~/.cache/huggingface/hub/` |

---

## Python Dependencies

```
mlx-lm              — MLX inference + LoRA training
chromadb            — Vector database สำหรับ RAG
sentence-transformers — Embedding model
pypdf               — อ่าน PDF
datasets            — โหลดและจัดการ dataset
pandas              — จัดการข้อมูลตาราง
scikit-learn        — ML utilities
transformers        — Tokenizer และ model utilities
accelerate          — Training acceleration
mcp                 — Model Context Protocol client/server
yfinance            — ดึงข้อมูลหุ้นจาก Yahoo Finance
```

---