"""
สคริปต์นี้สร้าง Vector Store สำหรับ RAG

📌 ค่า config ที่สำคัญ:
  • model_name="all-MiniLM-L6-v2" → โมเดลแปลงข้อความเป็น Vector
    - ถ้าค่ามาก: Vector แม่นยำ แต่ใช้ RAM มาก
    - ค่าน้อย: Vector ไม่แม่นยำ แต่ใช้ RAM น้อย
    - all-MiniLM-L6-v2 เป็นค่ามาตรฐานที่สมดุล

💡 ผลต่อ Inference:
  - Vector Store เก็บข้อความที่แปลงเป็น Vector
  - ตอนถามคำถาม → ข้อความถูกแปลงเป็น Vector → ค้นหาข้อความที่ใกล้เคียง
  - ถ้า Vector แม่นยำ → คำตอบแม่นยำ
  - ถ้า Vector ไม่แม่นยำ → คำตอบผิดเพี้ยน
"""

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./tor_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(
    name="tor_thai_passport", embedding_function=emb_fn
)


def ingest(text):
    """นำ text เข้า ChromaDB"""
    collection.add(documents=[text], ids=["id_0"])
    print(f"นำเข้า 1 document เรียบร้อย")


def query(query_text, n_results=1):
    """ค้นหาเอกสารที่เกี่ยวข้องจาก ChromaDB"""
    return collection.query(query_texts=[query_text], n_results=n_results)


if __name__ == "__main__":
    # ล้างข้อมูลเดิมถ้ามี
    if collection.count() > 0:
        client.delete_collection(name="tor_thai_passport")
        collection = client.get_or_create_collection(
            name="tor_thai_passport", embedding_function=emb_fn
        )

    # อ่าน PDF → ingest
    text = read_pdf("pdf/TOR-TH-AI-Pasport.pdf")
    ingest(text)

    # ทดสอบค้นหา
    print("\n--- ทดสอบค้นหา ---")
    for q in ["วัตถุประสงค์ของโครงการ", "จำนวนคนเป้าหมาย"]:
        results = query(q)
        print(f"\nคำค้นหา: {q}")
        print(f"  -> {results['documents'][0][0][:150]}...")

    print(f"\nจำนวน documents: {collection.count()}")