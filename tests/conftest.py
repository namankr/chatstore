"""
Shared pytest fixtures for chatstore tests.
Uses tmp_path so tests never write to the real workspace.
"""
import pytest
from chatstore import ChatService


@pytest.fixture()
def chat(tmp_path):
    """v1 ChatService backed by a temporary SQLite database."""
    return ChatService(project_id="test_project", db_path=str(tmp_path / "test.db"))


@pytest.fixture()
def chat_semantic(tmp_path):
    """v2 ChatService with semantic search, backed by a temp ChromaDB."""
    chromadb = pytest.importorskip("chromadb", reason="chromadb not installed — skipping semantic tests")
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed — skipping semantic tests")
    return ChatService(
        project_id="test_project",
        db_path=str(tmp_path / "test.db"),
        chroma_db_path=str(tmp_path / "chroma"),
        enable_semantic_search=True,
        embedding_model="all-MiniLM-L6-v2",
        memory_top_k=2,
        chunk_size=20,   # small chunks so tests don't need long text
    )
