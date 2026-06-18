"""
สคริปต์นี้แสดงการใช้งาน MCP (Model Context Protocol) + LLM ภาษาอีสาน

📌 ค่า config ที่สำคัญ:
  • adapter_path → ใช้โมเดลที่เทรนภาษาอีสานแล้ว
  • temp=0.6 → ความ "สร้างสรรค์" ของคำตอบ
  • top_p=0.9 → จำกัดเฉพาะ Token ที่มีโอกาสสะสม 90%

💡 ผลต่อ Inference:
  - MCP = โปรโตคอลที่เชื่อม LLM กับเครื่องมือภายนอก
  - LLM เรียกใช้เครื่องมือ → ดึงข้อมูล → วิเคราะห์ข้อมูล
  - โมเดลที่เทรนมาจะตอบเป็นภาษาอีสาน
"""

import asyncio
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mcp import ClientSession
from mcp.client.sse import sse_client

# โหลดโมเดลที่เทรนภาษาอีสานแล้ว
base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = str(Path(__file__).parent.parent / "02_finetuning" / "adapters")

model, tokenizer = load(base_model, adapter_path=adapter_path)
sampler = make_sampler(temp=0.6, top_p=0.9)

SERVER_URL = "http://localhost:8000/sse"


async def ask(query: str) -> str:
    # 1. เชื่อมต่อกับ MCP Server ผ่าน SSE
    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 2. ดึงข้อมูลดัชนี SET และหุ้นกลุ่ม SET50
            set_idx = (await session.call_tool("get_set_index", arguments={})).content[0].text
            top50 = (await session.call_tool("get_top_set50", arguments={})).content[0].text

            print("\n--- ข้อมูลที่ใช้วิเคราะห์ ---")
            print(set_idx)
            print(top50)
            print("----------------------------\n")

            # 3. สร้าง Prompt แบบ Chat Template (ภาษาอีสาน)
            messages = [
                {"role": "system", "content": "เจ้าเป็นนักวิเคราะห์หุ้นมืออาชีพ เจ้าต้องตอบเป็นภาษาอีสานเท่านั้น ตอบสั้นๆ ตรงประเด็น อ้างอิงข้อมูลที่ให้มาเท่านั้น"},
                {"role": "user", "content": f"""ข้อมูลตลาดหุ้นไทยวันนี้:
{set_idx}

หุ้นใน SET50:
{top50}

คำถาม: {query}

ตอบเป็นภาษาอีสาน:"""},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, max_tokens=512
            )

            # 4. สร้างคำตอบด้วย LLM
            raw = generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=200)
            return raw.split("<|")[0].strip()


if __name__ == "__main__":
    q = "วันนี้ตลาดหุ้นไทยเป็นจั่งใดบ้าง"
    print(f"คำถาม: {q}")
    print(f"คำตอบ: {asyncio.run(ask(q))}")
