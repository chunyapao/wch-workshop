"""
MCP Server ดึงข้อมูลหุ้นไทยและดัชนี SET ด้วย yfinance

📌 ค่า config ที่สำคัญ:
  • SET50_STOCKS → รายชื่อหุ้นที่จะดึงข้อมูล
    - ถ้าค่ามาก: ข้อมูลเยอะ แต่ใช้เวลานาน
    - ค่าน้อย: ข้อมูลน้อย แต่ใช้เวลาน้อย
    - 10 ตัวเป็นค่าเริ่มต้น

💡 ผลต่อ Inference:
  - Server ให้บริการ API สำหรับ LLM
  - LLM เรียกใช้เครื่องมือ → ดึงข้อมูล → วิเคราะห์ข้อมูล
  - ข้อมูลที่ดึงมา → ถูกส่งให้ LLM วิเคราะห์
  - ถ้าข้อมูลมีคุณภาพ → คำตอบแม่นยำ
  - ถ้าข้อมูลไม่มีคุณภาพ → คำตอบผิดเพี้ยน
"""
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# 1. สร้าง MCP Server Instance
mcp = FastMCP("set50")

# 2. กำหนดรายชื่อหุ้นใน SET50 (ตัวอย่าง)
SET50_STOCKS = [
    "PTT", "PTTEP", "PTTGC", "AOT", "ADVANC",
    "CPALL", "BDMS", "BBL", "KBANK", "SCB",
]


@mcp.tool()
def get_set_index() -> str:
    """ดึงข้อมูลดัชนี SET (^SET) ราคาล่าสุดจาก Yahoo Finance"""
    # 3. ดึงข้อมูลดัชนี SET
    t = yf.Ticker("^SET")
    info = t.fast_info
    hist = t.history(period="2d")

    # 4. คำนวณการเปลี่ยนแปลง
    last = info.last_price
    if len(hist) >= 2:
        prev = hist["Close"].iloc[-2]
        change = last - prev
        pct = (change / prev) * 100
    else:
        change = pct = 0.0

    return (
        f"ดัชนี SET: {last:,.2f} "
        f"เปลี่ยนแปลง {change:+.2f} ({pct:+.2f}%)"
    )


@mcp.tool()
def get_stock(symbol: str) -> str:
    """ดึงราคาหุ้นไทยรายตัว (ระบุชื่อโดยไม่ต้องเติม .BK)

    Args:
        symbol: ชื่อหุ้น เช่น PTT, KBANK, AOT
    """
    # 5. ดึงข้อมูลหุ้นรายตัว
    t = yf.Ticker(f"{symbol.upper()}.BK")
    hist = t.history(period="2d")

    if hist.empty:
        return f"ไม่พบข้อมูลหุ้น {symbol}"

    # 6. คำนวณการเปลี่ยนแปลง
    last = hist["Close"].iloc[-1]
    if len(hist) >= 2:
        prev = hist["Close"].iloc[-2]
        change = last - prev
        pct = (change / prev) * 100
    else:
        change = pct = 0.0

    return (
        f"{symbol.upper()}: {last:.2f} บาท "
        f"เปลี่ยนแปลง {change:+.2f} ({pct:+.2f}%)"
    )


@mcp.tool()
def get_top_set50() -> str:
    """ดึงราคาหุ้น SET50 กลุ่มตัวอย่าง 10 ตัว (ดึงพร้อมกันเป็นชุด)"""
    # 7. ดึงข้อมูลหุ้นหลายตัวพร้อมกัน
    tickers = [f"{s}.BK" for s in SET50_STOCKS]
    hist = yf.download(tickers, period="2d", auto_adjust=True, progress=False)

    # 8. แยกข้อมูลแต่ละหุ้น
    lines = []
    for sym in SET50_STOCKS:
        ticker = f"{sym}.BK"
        try:
            closes = hist["Close"][ticker].dropna()
            if closes.empty:
                lines.append(f"{sym:8s}  ไม่มีข้อมูล")
                continue
            last = closes.iloc[-1]
            pct = ((last - closes.iloc[-2]) / closes.iloc[-2] * 100) if len(closes) >= 2 else 0.0
            lines.append(f"{sym:8s}  {last:8.2f}  ({pct:+.2f}%)")
        except Exception:
            lines.append(f"{sym:8s}  ไม่มีข้อมูล")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
