"""
OCR fallback for scanned/image-only PDF pages.
 
Uses PyMuPDF (fitz) to render a PDF page to an image entirely in Python
(no Poppler dependency), then Tesseract OCR to read the text out of it.
 
Requires the Tesseract OCR binary to be installed separately:
  Windows: https://github.com/UB-Mannheim/tesseract/wiki
  (install, then note the install path, e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)
 
If Tesseract/pytesseract/pymupdf aren't available, OCR_AVAILABLE is False
and ingestion simply skips OCR (scanned pages are treated as empty, same
as before this feature existed).
"""
import os
 
OCR_AVAILABLE = True
 
try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    import io
 
    # If Tesseract isn't on PATH, point pytesseract at it explicitly via
    # this environment variable (set it before starting the server), e.g.:
    #   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
    _tesseract_cmd = os.getenv("TESSERACT_CMD")
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
 
except ImportError:
    OCR_AVAILABLE = False
 
 
def ocr_page_text(pdf_path: str, page_index: int, dpi: int = 200) -> str:
    """
    Render a single PDF page (0-indexed) to an image and run OCR on it.
    Returns an empty string if OCR isn't available or fails.
    """
    if not OCR_AVAILABLE:
        return ""
 
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        zoom = dpi / 72  # PDF default is 72 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        return pytesseract.image_to_string(img)
    except Exception as exc:
        print(f"OCR failed on page {page_index + 1}: {exc}")
        return ""
 