"""
Tests for VectorMemory / semantic search (v2).
All tests are auto-skipped if chromadb or sentence-transformers are not installed.
Run with:  pytest tests/test_memory.py -v
"""
import pytest


# ── store and retrieve ─────────────────────────────────────────────────────

class TestStoreAndRetrieve:
    def test_retrieve_returns_string(self, chat_semantic):
        chat_semantic.store_in_memory("Paris is the capital of France.", source="web_search")
        result = chat_semantic.retrieve_from_memory("capital of France")
        assert isinstance(result, str)

    def test_retrieve_returns_relevant_chunk(self, chat_semantic):
        chat_semantic.store_in_memory(
            "The Eiffel Tower is located in Paris France and is 330 metres tall.",
            source="web_search",
        )
        result = chat_semantic.retrieve_from_memory("Eiffel Tower height")
        assert result != ""

    def test_retrieve_returns_empty_when_nothing_stored(self, chat_semantic):
        result = chat_semantic.retrieve_from_memory("anything")
        assert result == ""

    def test_source_label_is_arbitrary(self, chat_semantic):
        """source= can be any string — the library stores it as metadata."""
        for source in ["pdf_parser", "database_query", "user_upload", "my_custom_tool"]:
            chat_semantic.store_in_memory(
                f"Some content from {source} " * 10, source=source
            )
        # retrieval should still work regardless of source labels
        result = chat_semantic.retrieve_from_memory("content")
        assert isinstance(result, str)


# ── session isolation ──────────────────────────────────────────────────────

class TestMemorySessionIsolation:
    def test_different_sessions_dont_share_memory(self, tmp_path):
        pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")
        from chatstore import ChatService

        def make(session_id=None):
            return ChatService(
                project_id="proj",
                db_path=str(tmp_path / "t.db"),
                chroma_db_path=str(tmp_path / "chroma"),
                enable_semantic_search=True,
                embedding_model="all-MiniLM-L6-v2",
                memory_top_k=2,
                chunk_size=20,
                session_id=session_id,
            )

        s1 = make()
        s2 = make()

        s1.store_in_memory("Secret data only for session one " * 5, source="test")

        # s2 has a different session_id so the filter should return nothing
        result = s2.retrieve_from_memory("Secret data session one")
        assert result == ""


# ── clear memory ───────────────────────────────────────────────────────────

class TestClearMemory:
    def test_clear_memory_removes_chunks(self, chat_semantic):
        chat_semantic.store_in_memory("Data to be deleted " * 10, source="test")
        chat_semantic.clear_memory()
        result = chat_semantic.retrieve_from_memory("Data to be deleted")
        assert result == ""

    def test_clear_memory_on_service_without_semantic_is_noop(self, chat):
        """Calling clear_memory on a v1 instance should not raise."""
        chat.clear_memory()   # no exception expected


# ── no-op behaviour when semantic search is disabled ──────────────────────

class TestSemanticDisabledNoop:
    def test_store_in_memory_is_noop_without_semantic(self, chat):
        chat.store_in_memory("anything", source="test")   # must not raise

    def test_retrieve_from_memory_returns_empty_without_semantic(self, chat):
        assert chat.retrieve_from_memory("anything") == ""
