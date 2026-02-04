from __future__ import annotations
from typing import List, Tuple
from pypdf import PdfReader
from core.logging import setup_logger
from rag.schemas import Chunk

logger = setup_logger("ingest")
SUPPORTED_EXTS = {".pdf", ".txt", ".md"}


def _read_pdf(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_index, page_text)
    """
    pages: List[Tuple[int, str]] = []
    reader = PdfReader(io_bytes(file_bytes))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
    logger.info(f"Parsed PDF: {filename} pages_with_text={len(pages)}")
    return pages

def io_bytes(data: bytes):
    import io
    return io.BytesIO(data)

def _read_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1600, overlap: int = 120) -> List[str]:
    """
    Simple character-based chunking (fast, reliable).
    chunk_size/overlap chosen to keep context small & retrieval effective.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)

    return chunks

def ingest_file(filename: str, file_bytes: bytes) -> List[Chunk]:
    """
    Convert a file into list of Chunk objects with metadata.
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        pages = _read_pdf(file_bytes, filename)
        all_chunks: List[Chunk] = []
        chunk_id = 0
        for page_idx, page_text in pages:
            parts = chunk_text(page_text)
            for part in parts:
                all_chunks.append(
                    Chunk(text=part, source=filename, chunk_id=chunk_id, page=page_idx)
                )
                chunk_id += 1
        logger.info(f"Ingested {filename}: chunks={len(all_chunks)}")
        return all_chunks

    if lower.endswith(".txt") or lower.endswith(".md"):
        text = _read_text(file_bytes)
        parts = chunk_text(text)
        chunks = [Chunk(text=p, source=filename, chunk_id=i) for i, p in enumerate(parts)]
        logger.info(f"Ingested {filename}: chunks={len(chunks)}")
        return chunks

    raise ValueError(f"Unsupported file type: {filename}")