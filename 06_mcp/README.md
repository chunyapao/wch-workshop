# Module 6 — MCP (Model Context Protocol)

เชื่อมต่อ LLM กับ MCP server เพื่อให้โมเดลเรียกใช้ external tools ได้แบบ realtime

---

## ไฟล์ในโมดูล

| ไฟล์ | คำอธิบาย |
|---|---|
| `01_inference.py` | ถามเวลาปัจจุบันผ่าน MCP time server |
| `02_inference.py` | วิเคราะห์หุ้นไทย SET50 ด้วยข้อมูล realtime จาก Yahoo Finance |
| `servers/time.py` | MCP server — ดึงเวลาท้องถิ่น / UTC |
| `servers/set50.py` | MCP server — ดึงดัชนี SET และราคาหุ้นไทย |

---

## วิธีรัน

```bash
uv run python 01_inference.py   # ถามเวลา
uv run python 02_inference.py   # วิเคราะห์หุ้น SET50
```

> **หมายเหตุ:** Module นี้ต้องการอินเทอร์เน็ตสำหรับดึงข้อมูลหุ้น realtime
