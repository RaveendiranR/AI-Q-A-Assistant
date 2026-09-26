"""
Embedding generation and FAISS vector store.
 
Uses a local Hugging Face embedding model and FAISS for semantic search.
The index is persisted to disk so it survives API restarts.
"""
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH = str(PROJECT_ROOT / "vectorstore" / "faiss_index")
 
_embedding_model = None  # cached singleton — loading this from disk is slow
 
 
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model
 
 
def build_vectorstore(chunks: list[dict], source_name: str) -> FAISS:
    """Convert page-aware chunks into LangChain Documents and index them."""
    embeddings = get_embedding_model()
 
    docs = [
        Document(
            page_content=item["text"],
            metadata={
                "source": source_name,
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                # FIX: carry the OCR flag through into vector metadata so
                # normal RAG/chat answers can show "• OCR" on their sources,
                # not just the dedicated word-count tool.
                "ocr": item.get("ocr", False),
            },
        )
        for item in chunks
    ]
 
    if not docs:
        raise ValueError("No text chunks were produced from the document.")
 
    return FAISS.from_documents(docs, embeddings)
 
 
def save_vectorstore(vectorstore: FAISS, path: str = VECTORSTORE_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(path)
 
 
def load_vectorstore(path: str = VECTORSTORE_PATH) -> FAISS:
    embeddings = get_embedding_model()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
 
 
def similarity_search(
    vectorstore: FAISS,
    query: str,
    k: int = 4,
) -> list[Document]:
    """Return the most relevant chunks for a query."""
    return vectorstore.similarity_search(query, k=k)
 