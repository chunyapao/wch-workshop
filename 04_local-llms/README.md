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
  -d '{"model": "default", "messages": [{"role": "user", "content": "สวัสดี"}]}'
```
