# Changelog

All notable changes to chatstore are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [2.0.0] - 2026-06-17

### Added
- `enable_semantic_search=True` flag on `ChatService` — activates ChromaDB vector memory
- `store_in_memory(text, source)` — chunk and embed any text (tool results, docs, etc.)
- `retrieve_from_memory(query)` — fetch semantically relevant past chunks per query
- `clear_memory()` — wipe vector memory for the current session
- `chatstore[semantic]` optional install extra (`chromadb`, `sentence-transformers`)
- Session-scoped retrieval — sessions never bleed into each other
- Configurable `embedding_model`, `memory_top_k`, `chunk_size`, `chroma_db_path`

---

## [1.0.0] - 2026-06-17

### Added
- `ChatService` — SQLite-backed persistent chat history
- `save_message(role, content)` — persist a message immediately
- `load_history(last_n)` — full or partial conversation history
- `load_history_windowed()` — sliding window of last N messages
- `build_context(extra_context)` — windowed history with optional RAG prefix
- `count_tokens(messages)` — approximate token count via tiktoken
- `clear_session()` — delete all messages for current session
- `get_session_id()` / `list_sessions()` — session utilities
- Multi-project and multi-session isolation via `project_id` and `session_id`
- `max_history_turns` and `max_context_tokens` fully configurable
- `tiktoken` as the only core dependency
