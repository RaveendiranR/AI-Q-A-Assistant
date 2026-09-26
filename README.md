# AI Document Q&A Assistant — Intermediate RAG Project

An intermediate-level Retrieval-Augmented Generation (RAG) project. Upload PDFs, ask
questions in plain English, and get answers grounded in the actual document content —
with sources, page numbers, and OCR fallback for scanned pages.

## Architecture

```text
PDF
 │
 ▼
FastAPI /upload
 │
 ▼
PyPDF (+ OCR fallback via Tesseract for scanned pages)
 │
 ▼
Page-aware chunking
 │
 ▼
Hugging Face Embeddings
 │
 ▼
FAISS Vector Database
 │
 ▼
Semantic Retrieval
 │
 ▼
Grounded Prompt
 │
 ▼
Ollama Local LLM
 │
 ▼
FastAPI /ask
 │
 ▼
HTML + CSS + JavaScript UI
```

## Technologies

- Python
- FastAPI
- PyPDF
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Ollama
- Tesseract OCR (via PyMuPDF + pytesseract)
- HTML
- CSS
- JavaScript

## Features

1. Multiple PDF support
2. Persistent document manifest
3. Page-aware chunk metadata
4. Source/page display in the frontend, including an OCR badge when a source page was scanned rather than text-extracted
5. Local LLM through Ollama
6. Grounded prompt to reduce unsupported answers
7. Upload size validation
8. PDF validation
9. Delete document and rebuild index
10. Health/status API
11. Separate frontend files
12. Responsive frontend
13. Loading/error states
14. Document statistics
15. Environment variables for LLM configuration
16. OCR fallback for scanned/image-only PDF pages (Tesseract)
17. Exact word/phrase counting answered directly from chat — e.g. "how many times does X appear?" — bypasses the LLM entirely for accuracy
18. Exact page counting answered directly from chat — e.g. "how many pages in the file?"
19. Fallback suggestions (Google/DuckDuckGo search + rephrase shortcut) shown in the UI whenever an answer isn't found in the uploaded documents

## Setup

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Install Ollama from the official Ollama website, then download a small local model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running. You can verify it with:

```bash
ollama list
```

### 4. (Optional) Install Tesseract for OCR support

Scanned/image-only PDF pages need Tesseract OCR installed separately for text extraction
to work on them. Without it, those pages are simply skipped — everything else still works.

Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, then point the app at
it if it's not on your PATH:

```bash
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 5. Start the FastAPI server

From the project root:

```bash
python -m uvicorn app.main:app --reload
```

API: `http://127.0.0.1:8000`
Swagger docs: `http://127.0.0.1:8000/docs`

### 6. Start the frontend

Open `frontend/index.html` in the browser.

If your browser blocks local file requests, use a simple static server:

```bash
cd frontend
python -m http.server 5500
```

Then open `http://127.0.0.1:5500`.

## Important

The first embedding operation downloads the `all-MiniLM-L6-v2` model and may take some time.

The first Ollama model request may also take longer because the model needs to load into memory.

## API endpoints

### GET /health
Checks API and index status.

### POST /upload
Uploads and indexes a PDF.

### GET /documents
Lists indexed documents.

### DELETE /documents/{filename}
Deletes a document and rebuilds the FAISS index from remaining PDFs.

### GET /documents/{filename}/word-count?word=...
Exact word/phrase count for one document, with per-page breakdown and OCR flag.

### POST /ask
Example:

```json
{
  "question": "What is the main topic of the document?",
  "k": 4
}
```

Also understands, without ever calling the LLM:
- "How many times does X appear?" / "how often does X appear?" — exact word/phrase count
- "How many pages in the file?" — exact page count, read directly from the PDF

Response contains:
- answer
- sources (filename, page, OCR flag, and a preview or occurrence count depending on the question type)

## Interview explanation

A simple explanation:

> "I built a document question-answering system using RAG. First, the application extracts text from uploaded PDFs and splits it into overlapping chunks, falling back to OCR for scanned pages. I generate embeddings for those chunks using a Hugging Face Sentence Transformer and store them in FAISS. When a user asks a question, the system retrieves the most semantically relevant chunks and sends only that context to a local Ollama LLM. Questions that ask for exact counts — like word occurrences or page counts — are detected and answered directly from the source text instead, since LLMs aren't reliable at counting. The answer is returned with source file, page, and OCR information, which makes the response traceable, and if nothing relevant is found, the UI offers to search the web or rephrase instead of dead-ending."

## Next possible upgrades

For a stronger portfolio version, you can later add:

- authentication
- PostgreSQL document metadata
- hybrid keyword + vector search
- reranking
- streaming LLM responses
- background document processing
- Docker
- automated tests
- deployment

## Windows folder setup — important

Do NOT copy this project over an old web project folder.

1. Create a new folder, for example: `C:\web\AI_QA_Assistant`
2. Extract this ZIP into that folder.
3. The folder should contain ONLY:
   `app`, `frontend`, `data`, `vectorstore`, `requirements.txt`,
   `.env.example`, `.gitignore`, `README.md`, and `run_backend.bat`.
4. The `frontend` folder must contain exactly:
   `index.html`, `style.css`, and `app.js`.

If you previously had files such as `index.html`, `quiz.css`, or `quiz.js`
in the project root, remove them or use a fresh folder. They are not part of
this project.
