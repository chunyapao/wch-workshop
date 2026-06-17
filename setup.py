#!/usr/bin/env python3
"""
setup.py — เตรียม environment สำหรับ WCH Workshop
รันครั้งเดียวขณะมีอินเทอร์เน็ต จากนั้นสามารถทำ workshop ได้แบบ offline
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

BASE_MODEL = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
EMBED_MODEL = "all-MiniLM-L6-v2"


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run(cmd: list[str], **kwargs):
    """รัน subprocess แล้ว raise ถ้าเกิด error"""
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def step_install_packages():
    banner("1/4  ติดตั้ง Python packages (uv sync)")
    run(["uv", "sync"], cwd=ROOT)
    print("✅ ติดตั้ง packages เสร็จสิ้น")


def step_download_llm():
    banner(f"2/4  ดาวน์โหลดโมเดล LLM — {BASE_MODEL}")
    print("(ไฟล์จะถูก cache ไว้ที่ ~/.cache/huggingface/hub/)")

    script = f"""
import sys
print("กำลังโหลดโมเดล {BASE_MODEL} ...")
from mlx_lm import load
model, tokenizer = load("{BASE_MODEL}")
del model, tokenizer
print("✅ โมเดล LLM พร้อมใช้งาน offline")
"""
    run(["uv", "run", "python", "-c", script], cwd=ROOT)


def step_download_embedding():
    banner(f"3/4  ดาวน์โหลด Embedding model — {EMBED_MODEL}")
    print("(ใช้ใน module 05_rag, ไฟล์จะถูก cache ไว้ที่ ~/.cache/huggingface/hub/)")

    script = f"""
print("กำลังโหลด Embedding model {EMBED_MODEL} ...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("{EMBED_MODEL}")
# ทดสอบ encode เพื่อให้แน่ใจว่าพร้อมใช้
_ = model.encode(["test warmup"])
del model
print("✅ Embedding model พร้อมใช้งาน offline")
"""
    run(["uv", "run", "python", "-c", script], cwd=ROOT)


def step_verify():
    banner("4/4  ตรวจสอบ imports ทั้งหมด")

    script = """
checks = [
    ("mlx_lm",              "from mlx_lm import load, generate"),
    ("mlx_lm.sample_utils", "from mlx_lm.sample_utils import make_sampler"),
    ("mlx.optimizers",      "import mlx.optimizers as optim"),
    ("chromadb",            "import chromadb"),
    ("sentence_transformers","from sentence_transformers import SentenceTransformer"),
    ("pypdf",               "from pypdf import PdfReader"),
    ("datasets",            "from datasets import Dataset"),
    ("sklearn",             "from sklearn.model_selection import train_test_split"),
    ("transformers",        "from transformers import AutoTokenizer"),
    ("mcp",                 "from mcp import ClientSession"),
    ("yfinance",            "import yfinance as yf"),
]
"""
    run(["uv", "run", "python", "-c", script], cwd=ROOT)


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║           WCH Workshop — Setup Script                     ║
║  รันสคริปต์นี้เพื่อเตรียม environment ให้พร้อมทำ workshop  ║
║  (ต้องการอินเทอร์เน็ตในการรันครั้งแรก)                    ║
╚════════════════════════════════════════════════════════════╝
""")

    try:
        step_install_packages()
        step_download_llm()
        step_download_embedding()
        step_verify()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ เกิดข้อผิดพลาดที่ขั้นตอน: {e}")
        sys.exit(1)

    print(f"""
{'='*60}
  ✅  Setup เสร็จสมบูรณ์! พร้อมทำ Workshop แบบ Offline แล้ว
{'='*60}

สิ่งที่ดาวน์โหลดและติดตั้งแล้ว:
  📦 Python packages  →  .venv/
  🤖 LLM หลัก        →  ~/.cache/huggingface/hub/
     {BASE_MODEL}
  🔍 Embedding model  →  ~/.cache/huggingface/hub/
     {EMBED_MODEL}

วิธีรัน Workshop แต่ละ Module:
  01 Pretraining      →  cd 01_pretraining      && uv run python 00_base-model.py
  02 Fine-tuning      →  cd 02_finetuning       && uv run python 01_training-run.py
  03 Prompt Eng.      →  cd 03_prompt-engineering && uv run python 01_zero-shot.py
  04 Local LLMs       →  cd 04_local-llms       && uv run python start-server.py
  05 RAG              →  cd 05_rag              && uv run python 03_inference.py
  06 MCP (เวลา)       →  cd 06_mcp              && uv run python 01_inference.py
  06 MCP (หุ้น SET50) →  cd 06_mcp              && uv run python 02_inference.py
""")


if __name__ == "__main__":
    main()
