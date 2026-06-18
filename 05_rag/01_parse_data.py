"""
สคริปต์นี้แยกข้อความจาก PDF สำหรับ RAG

📌 ค่า config ที่สำคัญ:
  • ไม่มีค่า config ที่สำคัญ → ใช้ค่าเริ่มต้นของ pypdf

💡 ผลต่อ Inference:
  - ข้อความที่แยกได้จะถูกแปลงเป็น Vector และเก็บใน ChromaDB
  - ถ้าข้อความมีคุณภาพสูง → คำตอบแม่นยำ
  - ถ้าข้อความมีเสียงรบกวน → คำตอบผิดเพี้ยน
  - PDF ที่มีหลายหน้า → ต้องแยกข้อความทุกหน้า
"""
from pypdf import PdfReader

def read_pdf(file_path):
    """อ่าน text จาก PDF"""
    # 1. เปิดไฟล์ PDF
    reader = PdfReader(file_path)
    
    # 2. แยกข้อความจากทุกหน้า
    text = "\n".join(page.extract_text() for page in reader.pages)
    
    # 3. ลบช่องว่างหัวท้ายแล้วส่งคืน
    return text.strip()


if __name__ == "__main__":
    # 4. ทดสอบการอ่าน PDF
    text = read_pdf("pdf/TOR-TH-AI-Pasport.pdf")
    print(f"จำนวนคำ: {len(text.split())}")
    print(text[:200] + "...")