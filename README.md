# chatstore

A lightweight, framework-agnostic persistent chat service for LLM applications.

No servers. No Docker. No cloud accounts. Just install and start building.

---

## The Problem It Solves

Every LLM application needs the same three things:

1. **Persist conversation history** across process restarts
2. **Feed a sliding window** of recent messages to the LLM (not the entire history)
3. **Optionally retrieve semantically relevant past context** using vector search

Most solutions force you into a framework (LangChain), require a running server (Zep), or call an LLM just to store a memory (Mem0).

`chatstore` does none of that. It is a single class, backed by SQLite, that works with any LLM provider — Gemini, OpenAI, Anthropic, Ollama, or your own model. Add semantic search later with one flag, no infrastructure changes.

---

## Install

**Core (SQLite + token counting):**
```bash
pip install chatstore
```

**With semantic search (adds ChromaDB + sentence-transformers):**
```bash
pip install chatstore[semantic]
```

---

## Quickstart

### v1 — Basic chat history

```python
from chatstore import ChatService

chat = ChatService(project_id="my_app")

# Save messages
chat.save_message("user", "What is the capital of France?")
chat.save_message("assistant", "The capital of France is Paris.")

# Load full history
history = chat.load_history()
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

# Build windowed context to pass to your LLM (last 10 turns by default)
context = chat.build_context()
```

### Resume an existing session

```python
session_id = chat.get_session_id()  # save this somewhere

# Later, in a new process:
chat = ChatService(project_id="my_app", session_id=session_id)
history = chat.load_history()  # full history is back
```

### Configure the sliding window size

The sliding window controls how many recent messages are passed to the LLM on each call. Adjust it based on your token budget and use case:

```python
# Default — last 10 messages
chat = ChatService(project_id="my_app")

# Larger window for complex, multi-turn conversations
chat = ChatService(project_id="my_app", max_history_turns=20)

# Smaller window for cost-sensitive or low-latency apps
chat = ChatService(project_id="my_app", max_history_turns=5)
```

> The full conversation history is always stored in SQLite regardless of window size.
> The window only controls what gets passed to the LLM — you can always call
> `load_history()` to retrieve the complete history.

### v2 — With semantic search

```python
from chatstore import ChatService

chat = ChatService(
    project_id="my_app",
    enable_semantic_search=True,
)

# After getting a tool/search result, store it in vector memory
# On source: it's just a free-text metadata label stored alongside the chunk. You can pass anything — "web_search", "pdf_parser", "database_query", "user_upload" — the library doesn't validate or care. It's only there so you can trace back where a chunk came from.

chat.store_in_memory("Paris is the capital of France. It has a population of 2.1 million.", source="web_search")

# Before calling your LLM, retrieve relevant past context
past_context = chat.retrieve_from_memory("What do we know about France?")

# Pass it into build_context
context = chat.build_context(extra_context=past_context)
# Now feed `context` to your LLM
```

---

## Integration Example (OpenAI)

```python
from openai import OpenAI
from chatstore import ChatService

client = OpenAI()
chat = ChatService(project_id="my_app")

def ask(user_input: str) -> str:
    chat.save_message("user", user_input)
    context = chat.build_context()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=context
    )
    reply = response.choices[0].message.content
    chat.save_message("assistant", reply)
    return reply
```

### Integration Example (Google Gemini)

```python
import google.generativeai as genai
from chatstore import ChatService

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-2.5-flash")
chat = ChatService(project_id="my_app")

def ask(user_input: str) -> str:
    chat.save_message("user", user_input)
    context = chat.build_context()

    # Convert to Gemini format
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in context
    ]
    response = model.generate_content(gemini_history)
    reply = response.text
    chat.save_message("assistant", reply)
    return reply
```

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `project_id` | required | Namespace — all sessions for one app share a project |
| `session_id` | auto UUID | Pass an existing ID to resume a session |
| `db_path` | `chat_history.db` | Path to the SQLite database file |
| `max_history_turns` | `10` | Sliding window size fed to the LLM |
| `max_context_tokens` | `8000` | Warns when context window approaches this limit |
| `enable_semantic_search` | `False` | Set `True` to enable v2 vector memory |
| `chroma_db_path` | `./chroma_db` | ChromaDB storage directory (v2 only) |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-Transformers model (v2 only) |
| `memory_top_k` | `3` | Number of chunks to retrieve per query (v2 only) |
| `chunk_size` | `400` | Words per memory chunk (v2 only) |

---

## Adapting for Production

`chatstore` ships with SQLite for zero-setup local use. As your application grows, you can swap components without changing your application code.

### Switch SQLite → PostgreSQL

The `_conn()` and `_init_db()` methods are the only two places that touch the database. Subclass `ChatService` and override them:

```python
import psycopg2
from chatstore import ChatService

class PostgresChatService(ChatService):
    def __init__(self, *args, dsn: str, **kwargs):
        self._dsn = dsn
        super().__init__(*args, **kwargs)

    def _conn(self):
        return psycopg2.connect(self._dsn)

    def _init_db(self):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id      SERIAL PRIMARY KEY,
                        project TEXT NOT NULL,
                        session TEXT NOT NULL,
                        role    TEXT NOT NULL,
                        content TEXT NOT NULL,
                        ts      TEXT NOT NULL
                    )
                """)
            conn.commit()

# Usage — identical to the base class
chat = PostgresChatService(
    project_id="my_app",
    dsn="postgresql://user:password@localhost:5432/mydb"
)
chat.save_message("user", "Hello!")
```

### Switch ChromaDB → Pinecone / Weaviate / Qdrant (v2)

Override `VectorMemory` in `chatstore/memory.py` — implement the same three methods (`store`, `retrieve`, `clear_session`) against your preferred vector DB:

```python
from chatstore.memory import VectorMemory
import pinecone

class PineconeChatMemory(VectorMemory):
    def __init__(self, project_id, api_key, index_name, **kwargs):
        # skip ChromaDB init entirely
        self._index = pinecone.Index(index_name)
        self._top_k = kwargs.get("top_k", 3)
        self._chunk_size = kwargs.get("chunk_size", 400)

    def store(self, text, source, session_id):
        # your Pinecone upsert logic here
        ...

    def retrieve(self, query, session_id):
        # your Pinecone query logic here
        ...
```

Then pass an instance directly to `ChatService._memory`:

```python
chat = ChatService(project_id="my_app")
chat._memory = PineconeChatMemory("my_app", api_key="...", index_name="chatstore")
```

### Use a custom embedding model

```python
chat = ChatService(
    project_id="my_app",
    enable_semantic_search=True,
    embedding_model="BAAI/bge-large-en-v1.5",   # any Sentence-Transformers model
)
```

### Multiple projects sharing one database

```python
# Each project_id is fully isolated — same db_path, zero bleed
support_chat = ChatService(project_id="support", db_path="shared.db")
sales_chat   = ChatService(project_id="sales",   db_path="shared.db")
```

---

## API Reference

```python
# Core (v1)
ChatService(project_id, session_id, db_path, max_history_turns, max_context_tokens)

chat.save_message(role: str, content: str)         # persist a message
chat.load_history(last_n=None) -> list[dict]        # full or partial history
chat.load_history_windowed()   -> list[dict]        # last max_history_turns messages
chat.build_context(extra_context="") -> list[dict]  # windowed history + optional prefix
chat.count_tokens(messages) -> int                  # approximate token count
chat.clear_session()                                # delete this session's messages
chat.get_session_id() -> str                        # current session UUID
chat.list_sessions() -> list[str]                   # all session IDs for this project

# Semantic search (v2, enable_semantic_search=True)
chat.store_in_memory(text: str, source: str)        # chunk and embed text
chat.retrieve_from_memory(query: str) -> str        # fetch relevant past chunks
chat.clear_memory()                                 # delete vector memory for session
```

---

## Roadmap

### v1.0 — Core ✅ Released
- SQLite-backed persistent chat history
- Sliding window context for LLM calls
- Token counting via tiktoken
- Multi-project and multi-session support

### v2.0 — Semantic Search ✅ Released
- ChromaDB-backed vector memory
- Local sentence-transformers embeddings (no API key required)
- Session-scoped retrieval — sessions never bleed into each other
- Optional install via `pip install chatstore[semantic]`

### v2.1 — Coming next
- **Async support** — `AsyncChatService` for FastAPI / async frameworks
- **PostgreSQL adapter** built-in (no subclassing required), enabled via `db_url="postgresql://..."`
- **Message metadata** — attach arbitrary `{key: value}` tags to any message for filtering
- **Session summaries** — auto-summarise old turns when approaching token limit instead of just dropping them

### v3.0 — Future
- **Built-in adapters** for Pinecone, Qdrant, and Weaviate (drop-in via `vector_backend="pinecone"`)
- **Multi-modal memory** — store and retrieve image descriptions alongside text
- **TTL / expiry** — auto-expire sessions older than N days
- **Export / import** — serialize a session to JSON for backup or migration

---

## License

MIT
