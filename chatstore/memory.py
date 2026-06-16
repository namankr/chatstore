import uuid

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Semantic search requires chromadb and sentence-transformers. "
        "Install them with: pip install chatstore[semantic]"
    ) from exc


def _chunk_text(text: str, source: str, session_id: str, chunk_size: int) -> list[dict]:
    """
    Split *text* into overlapping word-based chunks with metadata.
    Overlap = 20 % of chunk_size. Chunks shorter than 50 characters are dropped.
    """
    chunks = []
    words = text.split()
    step = max(1, int(chunk_size * 0.8))  # 20 % overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if len(chunk.strip()) < 50:
            continue
        chunks.append({
            "text": chunk,
            "metadata": {
                "source": source,
                "session_id": session_id,
                "chunk_index": len(chunks),
            },
        })
    return chunks


class VectorMemory:
    """
    ChromaDB-backed vector memory for a single project.

    Sessions are stored in a shared collection and filtered by ``session_id``
    metadata, so multiple conversations never bleed into each other.

    Parameters
    ----------
    project_id : str
        Name of the ChromaDB collection (prefixed with ``memory_``).
    chroma_db_path : str
        Directory where ChromaDB persists its data.
    embedding_model : str
        Sentence-Transformers model used for embeddings.
    top_k : int
        Number of chunks returned per ``retrieve`` call.
    chunk_size : int
        Words per chunk when storing text.
    """

    def __init__(
        self,
        project_id: str,
        chroma_db_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        chunk_size: int = 400,
    ):
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._client = chromadb.PersistentClient(path=chroma_db_path)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=f"memory_{project_id}",
            embedding_function=self._ef,
        )

    def store(self, text: str, source: str, session_id: str) -> None:
        """Chunk *text* and upsert into the collection."""
        chunks = _chunk_text(text, source, session_id, self._chunk_size)
        if not chunks:
            return
        self._collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[str(uuid.uuid4()) for _ in chunks],
        )

    def retrieve(self, query: str, session_id: str) -> str:
        """
        Return a formatted string of the top-k most relevant chunks for *query*,
        filtered to *session_id*. Returns an empty string when nothing is found.
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=self._top_k,
            where={"session_id": session_id},
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return "\n\n---\n".join(
            f"[Past finding {i + 1}]: {doc}" for i, doc in enumerate(docs)
        )

    def clear_session(self, session_id: str) -> None:
        """Delete all chunks belonging to *session_id*."""
        self._collection.delete(where={"session_id": session_id})
