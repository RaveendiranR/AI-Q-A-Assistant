import json
import os
import re
from pathlib import Path
 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
 
from app.ingest import ingest_pdf, load_pdf_pages
from app.embed_store import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    VECTORSTORE_PATH,
)
from app.rag_query import answer_question, OLLAMA_MODEL, OLLAMA_BASE_URL
 
 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
MANIFEST_PATH = DATA_DIR / "documents.json"
INDEX_BASE = VECTORSTORE_DIR / "faiss_index"
 
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
 
MAX_FILE_SIZE_MB = 25
 
 
app = FastAPI(
    title="AI Document Q&A Assistant",
    version="2.0.0",
    description="Intermediate RAG application using FastAPI, FAISS, Hugging Face embeddings and Ollama.",
)
 
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)
 
 
class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list
 
 
def safe_filename(filename: str) -> str:
    name = os.path.basename(filename or "")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "upload.pdf"
 
 
def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
 
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
 
 
def save_manifest(documents: list[dict]):
    MANIFEST_PATH.write_text(
        json.dumps(documents, indent=2),
        encoding="utf-8",
    )
 
 
def get_vectorstore():
    if Path(str(INDEX_BASE) + ".faiss").exists():
        return load_vectorstore(str(INDEX_BASE))
 
    return None
 
 
_vectorstore = get_vectorstore()
 
 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AI Document Q&A Assistant",
        "version": "2.0.0",
        "llm": OLLAMA_MODEL,
        "llm_url": OLLAMA_BASE_URL,
        "documents": len(load_manifest()),
        "index_ready": _vectorstore is not None,
    }
 
 
@app.get("/documents")
def list_documents():
    documents = load_manifest()
 
    return {
        "documents": documents,
        "count": len(documents),
    }
 
 
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global _vectorstore
 
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )
 
    filename = safe_filename(file.filename)
    save_path = DATA_DIR / filename
 
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    written = 0
 
    try:
        with save_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
 
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
                    )
 
                output.write(chunk)
 
    except HTTPException:
        if save_path.exists():
            save_path.unlink()
        raise
 
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save file: {exc}",
        )
 
    if written == 0:
        save_path.unlink(missing_ok=True)
 
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )
 
    try:
        chunks = ingest_pdf(str(save_path))
 
    except Exception as exc:
        save_path.unlink(missing_ok=True)
 
        raise HTTPException(
            status_code=422,
            detail=f"Could not read PDF: {exc}",
        )
 
    if not chunks:
        save_path.unlink(missing_ok=True)
 
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. This may be a scanned/image-only PDF.",
        )
 
    documents = load_manifest()
    is_replacement = any(
        d["filename"] == filename
        for d in documents
    )
 
    try:
        if is_replacement:
            documents = [
                d for d in documents
                if d["filename"] != filename
            ]
 
            documents.append({
                "filename": filename,
                "pages": len({
                    item["page"]
                    for item in chunks
                }),
                "chunks": len(chunks),
                "size_bytes": written,
            })
 
            combined_store = None
 
            for item in documents:
                path = DATA_DIR / item["filename"]
 
                if not path.exists():
                    continue
 
                remaining_chunks = ingest_pdf(str(path))
 
                if not remaining_chunks:
                    continue
 
                store = build_vectorstore(
                    remaining_chunks,
                    source_name=item["filename"],
                )
 
                if combined_store is None:
                    combined_store = store
                else:
                    combined_store.merge_from(store)
 
            _vectorstore = combined_store
 
        else:
            new_store = build_vectorstore(
                chunks,
                source_name=filename,
            )
 
            if _vectorstore is None:
                _vectorstore = new_store
            else:
                _vectorstore.merge_from(new_store)
 
            documents.append({
                "filename": filename,
                "pages": len({
                    item["page"]
                    for item in chunks
                }),
                "chunks": len(chunks),
                "size_bytes": written,
            })
 
        if _vectorstore is None:
            for suffix in (".faiss", ".pkl"):
                Path(
                    str(INDEX_BASE) + suffix
                ).unlink(missing_ok=True)
        else:
            save_vectorstore(
                _vectorstore,
                str(INDEX_BASE),
            )
 
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create/update vector index: {exc}",
        )
 
    save_manifest(documents)
 
    return {
        "filename": filename,
        "chunks_indexed": len(chunks),
        "pages": len({
            item["page"]
            for item in chunks
        }),
        "total_documents_indexed": len(documents),
        "message": "Document indexed successfully.",
    }
 
 
@app.delete("/documents/{filename}")
def delete_document(filename: str):
    global _vectorstore
 
    safe_name = safe_filename(filename)
    documents = load_manifest()
 
    if not any(
        d["filename"] == safe_name
        for d in documents
    ):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )
 
    remaining = [
        d for d in documents
        if d["filename"] != safe_name
    ]
 
    file_path = DATA_DIR / safe_name
 
    if file_path.exists():
        file_path.unlink()
 
    save_manifest(remaining)
 
    combined_store = None
 
    for item in remaining:
        path = DATA_DIR / item["filename"]
 
        if not path.exists():
            continue
 
        chunks = ingest_pdf(str(path))
 
        if not chunks:
            continue
 
        store = build_vectorstore(
            chunks,
            source_name=item["filename"],
        )
 
        if combined_store is None:
            combined_store = store
        else:
            combined_store.merge_from(store)
 
    _vectorstore = combined_store
 
    if _vectorstore is None:
        for suffix in (".faiss", ".pkl"):
            Path(
                str(INDEX_BASE) + suffix
            ).unlink(missing_ok=True)
    else:
        save_vectorstore(
            _vectorstore,
            str(INDEX_BASE),
        )
 
    return {
        "message": f"Deleted '{safe_name}'.",
        "remaining_documents": len(remaining),
    }
 
 
def is_count_question(question: str) -> bool:
    question_lower = question.lower()
 
    patterns = [
        r"\bhow many times\b",
        r"\bhow many occurrences\b",
        r"\bnumber of times\b",
        r"\bcount the word\b",
        r"\bcount the term\b",
        r"\bcount the phrase\b",
        r"\bcount how many\b",
        r"\bhow often\b",
        r"\bhow many\b.*\bappear\b",
        r"\bhow many\b.*\boccur\b",
    ]
 
    return any(
        re.search(pattern, question_lower)
        for pattern in patterns
    )
 
 
def extract_count_word(question: str) -> str | None:
    # FIX: the original patterns only recognized a leading "the word", so
    # phrasing like "the term X" or "the phrase X" leaked filler words
    # ("the term Python" instead of "Python") into the search term and
    # broke the count. Also added a dedicated "how often does X appear"
    # pattern, since is_count_question() already recognized "how often"
    # but extract_count_word() had no matching pattern for it, silently
    # falling back to the (slower, less precise) LLM path.
    filler = r"(?:the word|the term|the phrase)?\s*"
 
    patterns = [
        rf"how many times does\s+{filler}[\"']?(.+?)[\"']?\s+(?:appear|occur)\b",
        rf"how many times is\s+{filler}[\"']?(.+?)[\"']?\s+(?:mentioned|used)\b",
        rf"how often does\s+{filler}[\"']?(.+?)[\"']?\s+(?:appear|occur)\b",
        rf"how many occurrences of\s+{filler}[\"']?(.+?)[\"']?(?:\s+are there|\s+appear|\s+occur|\?|$)",
        rf"number of times\s+{filler}[\"']?(.+?)[\"']?\s+(?:appears|occurs)\b",
        r"count the (?:word|term|phrase)\s+[\"']?(.+?)[\"']?(?:\s+in|\?|$)",
    ]
 
    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            re.IGNORECASE,
        )
 
        if match:
            word = match.group(1).strip()
 
            word = re.sub(
                r"[?.!,]+$",
                "",
                word,
            ).strip()
 
            # Defensive cleanup in case a filler phrase still ended up
            # inside the captured group (covers "word"/"term"/"phrase"
            # with or without a leading "the").
            word = re.sub(
                r"^(?:the\s+)?(?:word|term|phrase)\s+",
                "",
                word,
                flags=re.IGNORECASE,
            ).strip()
 
            if word:
                return word
 
    return None
 
 
def is_page_count_question(question: str) -> bool:
    # A page-count question ("how many pages") is a different kind of
    # question from a word-count one ("how many times does X appear").
    # It can't be answered by the LLM at all: the LLM only ever sees a
    # handful of retrieved text chunks, not the whole document, and most
    # PDFs never state their own page count in the body text. This is
    # structural metadata we already have on disk, so we answer it
    # directly instead of asking the model.
    question_lower = question.lower()
 
    patterns = [
        r"\bhow many pages?\b",
        r"\bnumber of pages?\b",
        r"\bpage count\b",
        r"\bpages? (?:does|do|is|are)\b.*\b(?:have|has|contain|long)\b",
        r"\bhow long is\b.*\b(?:document|file|pdf|report|resume)\b",
    ]
 
    return any(
        re.search(pattern, question_lower)
        for pattern in patterns
    )
 
 
def get_page_counts() -> list[dict]:
    """True physical page count per uploaded PDF, read directly from the
    file rather than from the manifest (whose 'pages' field only counts
    pages that produced extractable/OCR'd text, and can undercount a PDF
    with a blank or failed-OCR page)."""
    documents = load_manifest()
 
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded.",
        )
 
    results = []
 
    for document in documents:
        filename = document["filename"]
        path = DATA_DIR / filename
 
        if not path.exists():
            continue
 
        try:
            total_pages = len(PdfReader(str(path)).pages)
        except Exception:
            # Fall back to the manifest's estimate if the file can't be
            # re-read for some reason, rather than failing the request.
            total_pages = document.get("pages")
 
        results.append({
            "filename": filename,
            "pages": total_pages,
        })
 
    return results
 
 
def count_word_in_documents(word: str) -> dict:
    documents = load_manifest()
 
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded.",
        )
 
    pattern = re.compile(
        r"\b" + re.escape(word) + r"\b",
        re.IGNORECASE,
    )
 
    total_count = 0
    page_results = []
 
    for document in documents:
        filename = document["filename"]
        path = DATA_DIR / filename
 
        if not path.exists():
            continue
 
        pages = load_pdf_pages(str(path))
 
        for page in pages:
            count = len(
                pattern.findall(page["text"])
            )
 
            if count:
                total_count += count
 
                page_results.append({
                    "source": filename,
                    "page": page["page"],
                    "count": count,
                    "ocr": page.get("ocr", False),
                    "preview": page["text"][:350],
                })
 
    return {
        "word": word,
        "total_count": total_count,
        "pages": page_results,
    }
 
 
@app.get("/documents/{filename}/word-count")
def word_count(filename: str, word: str):
    if not word.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide a word to count.",
        )
 
    safe_name = safe_filename(filename)
    path = DATA_DIR / safe_name
 
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )
 
    pages = load_pdf_pages(str(path))
 
    pattern = re.compile(
        r"\b" + re.escape(word.strip()) + r"\b",
        re.IGNORECASE,
    )
 
    per_page = []
    total = 0
 
    for page in pages:
        count = len(
            pattern.findall(page["text"])
        )
 
        if count:
            per_page.append({
                "page": page["page"],
                "count": count,
                "ocr": page.get("ocr", False),
            })
 
        total += count
 
    return {
        "filename": safe_name,
        "word": word,
        "total_count": total,
        "pages": per_page,
    }
 
 
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    question = request.question.strip()
 
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )
 
    if is_page_count_question(question):
        counts = get_page_counts()
 
        if len(counts) == 1:
            answer = (
                f'"{counts[0]["filename"]}" has '
                f'{counts[0]["pages"]} page(s).'
            )
        else:
            total = sum(c["pages"] for c in counts)
            breakdown = "; ".join(
                f'{c["filename"]}: {c["pages"]} page(s)'
                for c in counts
            )
            answer = (
                f"There are {total} page(s) total across "
                f"{len(counts)} documents ({breakdown})."
            )
 
        return {
            "question": question,
            "answer": answer,
            "sources": [],
        }
 
    if is_count_question(question):
        word = extract_count_word(question)
 
        if word:
            result = count_word_in_documents(word)
 
            if result["total_count"] == 0:
                answer = (
                    f'The word "{word}" was not found '
                    "in the uploaded documents."
                )
            else:
                answer = (
                    f'The word "{word}" appears '
                    f'{result["total_count"]} time(s) '
                    "in the uploaded documents."
                )
 
            return {
                "question": question,
                "answer": answer,
                "sources": result["pages"],
            }
 
    global _vectorstore
 
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
 
    if _vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No indexed documents. Upload a PDF first.",
        )
 
    try:
        return answer_question(
            _vectorstore,
            question,
            k=request.k,
        )
 
    except ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot connect to Ollama at "
                f"{OLLAMA_BASE_URL}. Start Ollama and make sure "
                f"model '{OLLAMA_MODEL}' is available."
            ),
        )
 
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {exc}",
        )
 
 
if __name__ == "__main__":
    import uvicorn
 
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
 