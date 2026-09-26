"""
RAG question-answering layer.
 
Retrieves relevant chunks from FAISS and sends only that context to a local
Ollama model. The prompt explicitly tells the model not to invent answers.
"""
import os
import requests
from langchain_core.documents import Document
from app.embed_store import similarity_search
 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
 
 
def build_context(documents: list[Document]) -> str:
    parts = []
 
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[SOURCE {i}] File: {source} | Page: {page}\n"
            f"{doc.page_content}"
        )
 
    return "\n\n".join(parts)
 
 
def call_ollama(question: str, context: str) -> str:
    prompt = f"""You are a document question-answering assistant.
 
Answer the user's question using ONLY the provided document context. Follow these rules:
 
1. If the question asks for a summary, overview, or "what is this document about" —
   synthesize a clear answer from the information actually present in the context.
   Do not refuse just because the question is broad or general.
2. If the question asks for a specific fact (a number, name, date, or detail) and
   that fact is genuinely absent from the context, say:
   "I couldn't find that information in the uploaded documents."
3. Do not invent facts that are not supported by the context.
4. Give a concise, clear answer. Don't restate the question.
5. When useful, mention the relevant source file and page.
6. Do not claim that you searched the internet.
 
DOCUMENT CONTEXT:
{context}
 
USER QUESTION:
{question}
"""
 
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(str(exc)) from exc
 
    data = response.json()
    return data.get("response", "").strip()
 
 
def answer_question(vectorstore, question: str, k: int = 4) -> dict:
    documents = similarity_search(vectorstore, question, k=k)
 
    if not documents:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": [],
        }
 
    context = build_context(documents)
    answer = call_ollama(question, context)
 
    sources = []
    seen = set()
 
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
 
        if key in seen:
            continue
 
        seen.add(key)
        sources.append({
            "source": source,
            "page": page,
            # FIX: propagate the OCR flag from vector metadata so the
            # frontend can show "• OCR" on normal chat answers too, not
            # just on word-count results.
            "ocr": doc.metadata.get("ocr", False),
            "preview": doc.page_content[:350],
        })
 
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
 