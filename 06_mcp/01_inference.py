"""MCP + MLX: ดึงข้อมูลหุ้นไทยด้วย yfinance แล้วให้ LLM วิเคราะห์"""

import asyncio, sys
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")
sampler = make_sampler(temp=0.6, top_p=0.9)

SERVER_PATH = str(Path(__file__).parent / "servers " / "set50.py")


async def ask(query: str) -> str:
    server = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ดึงดัชนี SET และหุ้นกลุ่ม SET50
            set_idx = (await session.call_tool("get_set_index", arguments={})).content[0].text
            top50 = (await session.call_tool("get_top_set50", arguments={})).content[0].text

            print("\n--- ข้อมูลที่ใช้วิเคราะห์ ---")
            print(set_idx)
            print(top50)
            print("----------------------------\n")

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

            raw = generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=150)
            return raw.split("<|")[0].strip()


if __name__ == "__main__":
    q = "วันนี้ตลาดหุ้นไทยเป็นอย่างไรบ้าง"
    print(f"คำถาม: {q}")
    print(f"คำตอบ: {asyncio.run(ask(q))}")
