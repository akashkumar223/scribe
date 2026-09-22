"""
Vector store wrapper around ChromaDB using a local sentence-transformers
embedding model (free, no API cost, works offline).
"""
import os

import uuid
import chromadb
from sentence_transformers import SentenceTransformer

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

_client = chromadb.PersistentClient(path=PERSIST_DIR)

# Load the model directly ourselves (not via chromadb's wrapper) so we can
# force local_files_only and guarantee no network call, even on restarts.
_model = None

def _get_model():
    """Load the embedding model on first use, not at import time — this lets
    the server bind its port immediately instead of blocking on a download."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", token=HF_TOKEN)
    return _model

class _LocalEmbeddingFunction:
    """Minimal embedding function matching chromadb's expected interface."""
    def __call__(self, input: list[str]) -> list[list[float]]:
        return _get_model().encode(input, convert_to_numpy=True).tolist()
_embed_fn = _LocalEmbeddingFunction()

_collection = _client.get_or_create_collection(
    name="documents",
    embedding_function=_embed_fn,
)


def add_chunks(doc_id: str, filename: str, chunks: list[str]) -> int:
    """Embed and store chunks for one document. Returns number of chunks stored."""
    if not chunks:
        return 0

    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "filename": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    _collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def query(question: str, n_results: int = 4) -> list[dict]:
    """Return the top-N most relevant chunks for a question, with metadata."""
    available = _collection.count()
    if not available:
        return []

    results = _collection.query(
        query_texts=[question],
        n_results=min(n_results, available),
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        hits.append({"text": doc, "filename": meta["filename"], "chunk_index": meta["chunk_index"]})
    return hits


def list_documents() -> list[str]:
    """Return distinct filenames currently stored."""
    all_meta = _collection.get()["metadatas"]
    return sorted({m["filename"] for m in all_meta}) if all_meta else []


def delete_document(filename: str) -> int:
    """Delete all chunks belonging to a filename. Returns number of chunks deleted."""
    existing = _collection.get(where={"filename": filename})
    ids_to_delete = existing.get("ids", [])
    if ids_to_delete:
        _collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def new_doc_id() -> str:
    return str(uuid.uuid4())[:8]
