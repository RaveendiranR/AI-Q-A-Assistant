# AI Document Q&A Assistant — Intermediate RAG Project

An intermediate-level Retrieval-Augmented Generation (RAG) project.

## Architecture

```text
PDF
 │
 ▼
FastAPI /upload
 │
 ▼
PyPDF
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
- HTML
- CSS
- JavaScript

## Features added compared with the beginner version

1. Multiple PDF support
2. Persistent document manifest
3. Page-aware chunk metadata
4. Source/page display in the frontend
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

Make sure Ollama is running.

You can verify it with:

```bash
ollama list
```

### 4. Start the FastAPI server

From the project root:

```bash
python -m uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the frontend

Open `frontend/index.html` in the browser.

If your browser blocks local file requests, use a simple static server:

```bash
cd frontend
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500
```

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

### POST /ask

Example:

```json
{
  "question": "What is the main topic of the document?",
  "k": 4
}
```

Response contains:

- answer
- source filename
- page number
- source preview

## Interview explanation

A simple explanation:

> "I built a document question-answering system using RAG. First, the application extracts text from uploaded PDFs and splits it into overlapping chunks. I generate embeddings for those chunks using a Hugging Face Sentence Transformer and store them in FAISS. When a user asks a question, the system retrieves the most semantically relevant chunks and sends only that context to a local Ollama LLM. The answer is returned with source file and page information, which helps make the response traceable."

## Next possible upgrades

For a stronger portfolio version, you can later add:

- authentication
- PostgreSQL document metadata
- hybrid keyword + vector search
- reranking
- streaming LLM responses
- background document processing
- OCR for scanned PDFs
- Docker
- automated tests
- deployment


## Windows folder setup — important

Do NOT copy this project over an old web project folder.

1. Create a new folder, for example:
   `C:\web\AI_QA_Assistant`
2. Extract this ZIP into that folder.
3. The folder should contain ONLY:
   `app`, `frontend`, `data`, `vectorstore`, `requirements.txt`,
   `.env.example`, `.gitignore`, `README.md`, and `run_backend.bat`.
4. The `frontend` folder must contain exactly:
   `index.html`, `style.css`, and `app.js`.

If you previously had files such as `index.html`, `quiz.css`, or `quiz.js`
in the project root, remove them or use a fresh folder. They are not part of
this project.
