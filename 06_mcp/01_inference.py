"""
สคริปต์นี้แสดงการใช้งาน MCP (Model Context Protocol) + LLM

📌 ค่า config ที่สำคัญ:
  • temp=0.6 → ความ “สร้างสรรค์” ของคำตอบ
    - ถ้าค่าน้อย (0.1): คำตอบแน่นอน แต่ซ้ำซาก
    - ถ้าค่ามาก (1.0): คำตอบสร้างสรรค์ แต่อาจเพี้ยน
    - 0.6 เป็นค่ากลางๆ ที่สมดุล

  • top_p=0.9 → จำกัดเฉพาะ Token ที่มีโอกาสสะสม 90%
    - ถ้าค่าน้อย: คำตอบแคบ แต่แน่นอน
    - ถ้าค่ามาก: คำตอบกว้าง แต่อาจเพี้ยน
    - 0.9 เป็นค่ามาตรฐาน

💡 ผลต่อ Inference:
  - MCP = โปรโตคอลที่เชื่อม LLM กับเครื่องมือภายนอก
  - LLM เรียกใช้เครื่องมือ → ดึงข้อมูล → วิเคราะห์ข้อมูล
  - temp/top_p ที่เหมาะสม → คำตอบสมดุลระหว่างสร้างสรรค์และแน่นอน
  - ใช้ Chat Template → คำตอบเป็นรูปแบบบทสนทนา
"""

import asyncio, sys
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mcp import ClientSession
from mcp.client.sse import sse_client

model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")
sampler = make_sampler(temp=0.6, top_p=0.9)

SERVER_URL = "http://localhost:8000/sse"


async def ask(query: str) -> str:
    # 1. เชื่อมต่อกับ MCP Server ผ่าน SSE
    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 3. ดึงข้อมูลดัชนี SET และหุ้นกลุ่ม SET50
            set_idx = (await session.call_tool("get_set_index", arguments={})).content[0].text
            top50 = (await session.call_tool("get_top_set50", arguments={})).content[0].text

            print("\n--- ข้อมูลที่ใช้วิเคราะห์ ---")
            print(set_idx)
            print(top50)
            print("----------------------------\n")

            # 4. สร้าง Prompt แบบ Chat Template
            messages = [
                {"role": "system", "content": "คุณเป็นนักวิเคราะห์หุ้นมืออาชีพ ตอบสั้นๆ ตรงประเด็น อ้างอิงข้อมูลที่ให้มาเท่านั้น"},
                {"role": "user", "content": f"""ข้อมูลตลาดหุ้นไทยวันนี้:
{set_idx}

หุ้นใน SET50:
{top50}

คำถาม: {query}"""},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 5. สร้างคำตอบด้วย LLM
            raw = generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=150)
            return raw.split("<|")[0].strip()


if __name__ == "__main__":
    q = "วันนี้ตลาดหุ้นไทยเป็นอย่างไรบ้าง"
    print(f"คำถาม: {q}")
    print(f"คำตอบ: {asyncio.run(ask(q))}")
