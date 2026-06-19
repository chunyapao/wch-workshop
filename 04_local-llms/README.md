# Module 4 — Local LLM Server

รัน LLM เป็น OpenAI-compatible API server บนเครื่องตัวเอง

---

## วิธีรัน

```bash
uv run python start-server.py
```

Server จะพร้อมที่ `http://localhost:8080`

---

## ทดสอบ API

เรียกใช้เหมือน OpenAI API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit", "messages": [{"role": "user", "content": "สวัสดี"}]}'  | jq -r '.choices[0].message.content'
```
**หมายเหตุ:** ใช้ `jq -r` เพื่อแยกเฉพาะเนื้อหาคำตอบ (ไม่ต้องอ่าน JSON ทั้งหมด)
