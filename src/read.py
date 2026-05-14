from docx import Document
import json
from pathlib import Path


# Các từ khóa nhận biết heading / tiêu đề mục
HEADING_KEYWORDS = [
    "chương", "điều", "mục", "phần", "quy định",
    "chapter", "section", "article"
]


def is_heading(text: str) -> bool:
    """Kiểm tra đoạn văn có phải tiêu đề mục không."""
    lower = text.lower().strip()
    return any(lower.startswith(kw) for kw in HEADING_KEYWORDS)


def read_docx_by_paragraph(file_path: str):
    """
    Đọc file .docx và trả về danh sách các đoạn văn.
    Mỗi đoạn văn = 1 chunk.
    """
    doc = Document(file_path)

    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()

        # Bỏ qua đoạn rỗng
        if text:
            paragraphs.append(text)

    return paragraphs


def build_chunks(file_path: str):
    """
    Chuyển nội dung file docx thành format dùng cho Naive RAG.
    Tự động detect section dựa trên heading keyword.
    """
    file_path = Path(file_path)
    paragraphs = read_docx_by_paragraph(str(file_path))

    chunks = []
    current_section = "Chung"  # section mặc định nếu chưa gặp heading

    for index, paragraph in enumerate(paragraphs, start=1):

        # Cập nhật section nếu đây là dòng heading
        if is_heading(paragraph):
            current_section = paragraph

        chunk = {
            "id": f"{file_path.stem}_chunk_{index:03d}",
            "text": paragraph,
            "metadata": {
                "source": file_path.name,
                "type": "school_rules",
                "chunk_index": index,
                "section": current_section  # FIX: thêm section để build_context dùng được
            }
        }

        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    input_file = "/content/naive_rag/data/1.4. Annex 4 - Internship Agreement_2026.docx"

    chunks = build_chunks(input_file)

    with open("school_rules_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Đã tạo {len(chunks)} chunks")
    print("File output: school_rules_chunks.json")