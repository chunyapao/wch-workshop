# Module 6 — MCP (Model Context Protocol)

เชื่อมต่อ LLM กับ MCP server เพื่อให้โมเดลเรียกใช้ external tools ได้แบบ realtime

ใช้ SSE (Server-Sent Events) สำหรับ communication ระหว่าง client และ server

---

## ไฟล์ในโมดูล

| ไฟล์ | คำอธิบาย |
|---|---|
| `01_inference.py` | วิเคราะห์หุ้นไทย SET50 ด้วยข้อมูล realtime |
| `01_inference-isan.py` | วิเคราะห์หุ้นไทย ตอบเป็นภาษาอีสาน (ใช้ adapter ที่เทรนแล้ว) |
| `servers/set50.py` | MCP server — ดึงดัชนี SET และราคาหุ้นไทย (SSE transport) |

---

## วิธีรัน

### Terminal 1: เปิด MCP Server
```bash
cd "servers " && uv run python set50.py
```
Server จะพร้อมที่ `http://localhost:8000`

### Terminal 2: รัน Inference
```bash
# ตอบภาษาไทย
uv run python 01_inference.py

# ตอบภาษาอีสาน (ใช้ adapter ที่เทรนแล้ว)
uv run python 01_inference-isan.py
```

> **หมายเหตุ:** Module นี้ต้องการอินเทอร์เน็ตสำหรับดึงข้อมูลหุ้น realtime
