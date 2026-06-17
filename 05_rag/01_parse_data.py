from pypdf import PdfReader

def read_pdf(file_path):
    """อ่าน text จาก PDF"""
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages)
    return text.strip()


if __name__ == "__main__":
    text = read_pdf("pdf/TOR-TH-AI-Pasport.pdf")
    print(f"จำนวนคำ: {len(text.split())}")
    print(text[:200] + "...")