"""
PDF ingestion pipeline.
 
Extracts page-aware text and creates overlapping chunks while preserving
page/source metadata. This makes source citations possible in the UI.
 
Pages with no extractable text (scanned/image PDFs) automatically fall
back to OCR via Tesseract, if it's installed.
"""
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from app.ocr import ocr_page_text, OCR_AVAILABLE
 
 
def load_pdf_pages(pdf_path: str) -> list[dict]:
    """Extract text page-by-page so each chunk can retain its page number.
    Falls back to OCR for any page with no real text layer."""
    reader = PdfReader(pdf_path)
    pages = []
 
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
 
        used_ocr = False
        if not text and OCR_AVAILABLE:
            text = ocr_page_text(pdf_path, page_num - 1).strip()
            used_ocr = True
 
        if text:
            pages.append({
                "page": page_num,
                "text": text,
                "ocr": used_ocr,
            })
 
    return pages
 
 
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Split each page into chunks and preserve page + OCR metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
 
    chunks = []
    chunk_id = 0
 
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for text in page_chunks:
            if text.strip():
                chunks.append({
                    "text": text.strip(),
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    # FIX (was missing before): without this, every chunk
                    # forgets whether its page came from OCR, so nothing
                    # downstream could ever show an OCR badge.
                    "ocr": page.get("ocr", False),
                })
                chunk_id += 1
 
    return chunks
 
 
def ingest_pdf(pdf_path: str) -> list[dict]:
    """Full pipeline: PDF -> page-aware chunks."""
    pages = load_pdf_pages(pdf_path)
 
    if not pages:
        return []
 
    chunks = chunk_pages(pages)
 
    print(
        f"Loaded '{Path(pdf_path).name}': "
        f"{len(pages)} pages -> {len(chunks)} chunks"
    )
    return chunks
 
 
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingest <path_to_pdf>")
        raise SystemExit(1)
 
    result = ingest_pdf(sys.argv[1])
    print(f"Created {len(result)} chunks")
    if result:
        print(result[0]["text"][:500])
 