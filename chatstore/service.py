import sqlite3
import uuid
from datetime import datetime
from typing import Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class ChatService:
    """
    Lightweight, persistent chat service.

    v1 (default) — SQLite-backed conversation history with a sliding window
    and optional token counting. Zero heavy dependencies.

    v2 (semantic search) — Pass ``enable_semantic_search=True`` to layer on
    ChromaDB vector memory so you can retrieve relevant past context per query.
    Requires the ``[semantic]`` extra: ``pip install chatstore[semantic]``

    Parameters
    ----------
    project_id : str
        Namespace for all sessions belonging to one project/app.
    session_id : str, optional
        Resume an existing session. Auto-generated UUID if omitted.
    db_path : str
        Path to the SQLite database file. Defaults to ``chat_history.db``.
    max_history_turns : int
        Sliding-window size (number of messages) fed to the LLM. Default 10.
    max_context_tokens : int
        Warn when the windowed context exceeds this token count. Default 8000.

    --- v2 optional semantic search ---
    enable_semantic_search : bool
        Set ``True`` to enable vector memory. Requires ``chatstore[semantic]``.
    chroma_db_path : str
        Directory for the ChromaDB persistent store. Default ``./chroma_db``.
    embedding_model : str
        Sentence-Transformers model name. Default ``all-MiniLM-L6-v2``.
    memory_top_k : int
        Number of chunks to retrieve per query. Default 3.
    chunk_size : int
        Words per memory chunk when storing tool results. Default 400.
    """

    def __init__(
        self,
        project_id: str,
        session_id: Optional[str] = None,
        db_path: str = "chat_history.db",
        max_history_turns: int = 10,
        max_context_tokens: int = 8000,
        # v2 ──────────────────────────────────────────────────────
        enable_semantic_search: bool = False,
        chroma_db_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        memory_top_k: int = 3,
        chunk_size: int = 400,
    ):
        self.project_id = project_id
        self.session_id = session_id or str(uuid.uuid4())
        self._db_path = db_path
        self._max_history_turns = max_history_turns
        self._max_context_tokens = max_context_tokens
        self._init_db()

        # v2: optional vector memory — lazily imported so v1 stays lightweight
        self._memory = None
        if enable_semantic_search:
            from chatstore.memory import VectorMemory  # noqa: PLC0415
            self._memory = VectorMemory(
                project_id=project_id,
                chroma_db_path=chroma_db_path,
                embedding_model=embedding_model,
                top_k=memory_top_k,
                chunk_size=chunk_size,
            )

    # ── DB bootstrap ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    project  TEXT NOT NULL,
                    session  TEXT NOT NULL,
                    role     TEXT NOT NULL,
                    content  TEXT NOT NULL,
                    ts       TEXT NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ── Core message operations ───────────────────────────────────────────

    def save_message(self, role: str, content: str) -> None:
        """Persist a single message immediately."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (project, session, role, content, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.project_id, self.session_id, role, content,
                 datetime.utcnow().isoformat()),
            )

    def load_history(self, last_n: Optional[int] = None) -> list[dict]:
        """
        Return conversation history as ``[{role, content}, ...]``.

        Parameters
        ----------
        last_n : int, optional
            Return only the most recent *last_n* messages. ``None`` = full history.
        """
        with self._conn() as conn:
            if last_n:
                rows = conn.execute("""
                    SELECT role, content FROM (
                        SELECT role, content, ts FROM messages
                        WHERE project=? AND session=?
                        ORDER BY ts DESC LIMIT ?
                    ) ORDER BY ts ASC
                """, (self.project_id, self.session_id, last_n)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content FROM messages "
                    "WHERE project=? AND session=? ORDER BY ts ASC",
                    (self.project_id, self.session_id),
                ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def clear_session(self) -> None:
        """Delete all messages for the current session."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM messages WHERE project=? AND session=?",
                (self.project_id, self.session_id),
            )

    # ── Windowed context ─────────────────────────────────────────────────

    def load_history_windowed(self) -> list[dict]:
        """Return the last ``max_history_turns`` messages."""
        return self.load_history(last_n=self._max_history_turns)

    def count_tokens(self, messages: list[dict]) -> int:
        """
        Approximate token count for a list of ``{role, content}`` messages.
        Uses tiktoken's ``cl100k_base`` encoder. Returns 0 if tiktoken is not
        installed.
        """
        if not _TIKTOKEN_AVAILABLE:
            return 0
        enc = tiktoken.get_encoding("cl100k_base")
        total = sum(len(enc.encode(m.get("content", ""))) + 4 for m in messages)
        return total

    def build_context(self, extra_context: str = "") -> list[dict]:
        """
        Build the message list to pass to an LLM:

        1. Sliding window of recent history.
        2. Optionally prepend ``extra_context`` (e.g. RAG results) as a
           system-style user message.

        Logs a warning when approaching ``max_context_tokens``.
        """
        history = self.load_history_windowed()
        token_count = self.count_tokens(history)

        if token_count and token_count > self._max_context_tokens * 0.85:
            print(
                f"[chatstore] ⚠️  Approaching token limit "
                f"({token_count}/{self._max_context_tokens})"
            )

        if extra_context:
            history = [{
                "role": "user",
                "content": f"[Relevant past findings for context]\n{extra_context}",
            }] + history

        return history

    # ── v2: semantic memory helpers ───────────────────────────────────────

    def store_in_memory(self, text: str, source: str) -> None:
        """
        (v2) Chunk and store *text* in vector memory under the current session.
        No-op if semantic search was not enabled.
        """
        if self._memory:
            self._memory.store(text, source, self.session_id)

    def retrieve_from_memory(self, query: str) -> str:
        """
        (v2) Retrieve the most relevant past chunks for *query*.
        Returns an empty string if semantic search was not enabled or nothing
        is found.
        """
        if self._memory:
            return self._memory.retrieve(query, self.session_id)
        return ""

    def clear_memory(self) -> None:
        """(v2) Remove all vector-memory entries for the current session."""
        if self._memory:
            self._memory.clear_session(self.session_id)

    # ── Utility ───────────────────────────────────────────────────────────

    def get_session_id(self) -> str:
        return self.session_id

    def list_sessions(self) -> list[str]:
        """Return all session IDs that exist for this project."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT session FROM messages "
                "WHERE project=? GROUP BY session ORDER BY MIN(ts)",
                (self.project_id,),
            ).fetchall()
        return [r[0] for r in rows]
