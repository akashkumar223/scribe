"""Parse PDF/DOCX/TXT files into plain text, then chunk for embedding."""
import io
from pypdf import PdfReader
from docx import Document as DocxDocument


def parse_file(filename: str, content: bytes) -> str:
    """Extract plain text from an uploaded file's bytes."""
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    if ext == "txt":
        return content.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks (by characters, simple + reliable for a hackathon).
    Overlap keeps context from being cut off between chunks.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
