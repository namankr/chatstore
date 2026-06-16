"""
Tests for ChatService core (v1) — no semantic search required.
Run with:  pytest tests/test_service.py -v
"""
import pytest
from chatstore import ChatService


# ── save / load ────────────────────────────────────────────────────────────

class TestSaveAndLoad:
    def test_save_single_message(self, chat):
        chat.save_message("user", "Hello")
        history = chat.load_history()
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "Hello"}

    def test_save_multiple_messages_preserves_order(self, chat):
        chat.save_message("user", "First")
        chat.save_message("assistant", "Second")
        chat.save_message("user", "Third")
        history = chat.load_history()
        assert [m["content"] for m in history] == ["First", "Second", "Third"]

    def test_load_history_returns_empty_list_for_new_session(self, chat):
        assert chat.load_history() == []

    def test_roles_are_preserved(self, chat):
        chat.save_message("user", "Hi")
        chat.save_message("assistant", "Hello!")
        history = chat.load_history()
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


# ── windowed history ───────────────────────────────────────────────────────

class TestWindowedHistory:
    def test_last_n_returns_correct_number(self, chat):
        for i in range(15):
            chat.save_message("user", f"msg {i}")
        history = chat.load_history(last_n=5)
        assert len(history) == 5

    def test_last_n_returns_most_recent(self, chat):
        for i in range(5):
            chat.save_message("user", f"msg {i}")
        history = chat.load_history(last_n=2)
        assert history[0]["content"] == "msg 3"
        assert history[1]["content"] == "msg 4"

    def test_windowed_respects_max_history_turns(self, tmp_path):
        chat = ChatService(
            project_id="p",
            db_path=str(tmp_path / "t.db"),
            max_history_turns=3,
        )
        for i in range(10):
            chat.save_message("user", f"msg {i}")
        assert len(chat.load_history_windowed()) == 3

    def test_last_n_none_returns_full_history(self, chat):
        for i in range(20):
            chat.save_message("user", f"msg {i}")
        assert len(chat.load_history()) == 20


# ── session isolation ──────────────────────────────────────────────────────

class TestSessionIsolation:
    def test_two_sessions_dont_share_messages(self, tmp_path):
        db = str(tmp_path / "shared.db")
        s1 = ChatService(project_id="proj", db_path=db)
        s2 = ChatService(project_id="proj", db_path=db)
        s1.save_message("user", "session one")
        assert s2.load_history() == []

    def test_resume_session_loads_history(self, tmp_path):
        db = str(tmp_path / "shared.db")
        s1 = ChatService(project_id="proj", db_path=db)
        s1.save_message("user", "persisted message")
        session_id = s1.get_session_id()

        s2 = ChatService(project_id="proj", db_path=db, session_id=session_id)
        history = s2.load_history()
        assert len(history) == 1
        assert history[0]["content"] == "persisted message"


# ── project isolation ──────────────────────────────────────────────────────

class TestProjectIsolation:
    def test_two_projects_dont_share_messages(self, tmp_path):
        db = str(tmp_path / "shared.db")
        p1 = ChatService(project_id="alpha", db_path=db)
        p2 = ChatService(project_id="beta",  db_path=db)
        p1.save_message("user", "alpha message")
        assert p2.load_history() == []

    def test_list_sessions_scoped_to_project(self, tmp_path):
        db = str(tmp_path / "shared.db")
        p1 = ChatService(project_id="alpha", db_path=db)
        p2 = ChatService(project_id="beta",  db_path=db)
        p1.save_message("user", "hi")
        p2.save_message("user", "hi")
        assert p1.get_session_id() not in p2.list_sessions()
        assert p2.get_session_id() not in p1.list_sessions()


# ── clear session ──────────────────────────────────────────────────────────

class TestClearSession:
    def test_clear_removes_all_messages(self, chat):
        chat.save_message("user", "to be deleted")
        chat.clear_session()
        assert chat.load_history() == []

    def test_clear_does_not_affect_other_sessions(self, tmp_path):
        db = str(tmp_path / "shared.db")
        s1 = ChatService(project_id="proj", db_path=db)
        s2 = ChatService(project_id="proj", db_path=db)
        s1.save_message("user", "keep me")
        s2.save_message("user", "delete me")
        s2.clear_session()
        assert len(s1.load_history()) == 1


# ── build_context ──────────────────────────────────────────────────────────

class TestBuildContext:
    def test_build_context_returns_list_of_dicts(self, chat):
        chat.save_message("user", "hello")
        context = chat.build_context()
        assert isinstance(context, list)
        assert all("role" in m and "content" in m for m in context)

    def test_build_context_prepends_extra_context(self, chat):
        chat.save_message("user", "hello")
        context = chat.build_context(extra_context="some background info")
        assert context[0]["role"] == "user"
        assert "some background info" in context[0]["content"]

    def test_build_context_without_extra_context_has_no_prefix(self, chat):
        chat.save_message("user", "hello")
        context = chat.build_context()
        assert "background" not in context[0]["content"]


# ── token counting ─────────────────────────────────────────────────────────

class TestTokenCounting:
    def test_count_tokens_returns_int(self, chat):
        messages = [{"role": "user", "content": "hello world"}]
        result = chat.count_tokens(messages)
        assert isinstance(result, int)

    def test_count_tokens_returns_zero_for_empty(self, chat):
        assert chat.count_tokens([]) == 0

    def test_count_tokens_increases_with_more_content(self, chat):
        short = [{"role": "user", "content": "hi"}]
        long  = [{"role": "user", "content": "hi " * 200}]
        assert chat.count_tokens(long) > chat.count_tokens(short)


# ── utility ────────────────────────────────────────────────────────────────

class TestUtility:
    def test_get_session_id_returns_string(self, chat):
        assert isinstance(chat.get_session_id(), str)

    def test_list_sessions_includes_current(self, chat):
        chat.save_message("user", "hi")   # must write at least one message
        assert chat.get_session_id() in chat.list_sessions()

    def test_two_instances_have_different_session_ids(self, tmp_path):
        db = str(tmp_path / "shared.db")
        s1 = ChatService(project_id="proj", db_path=db)
        s2 = ChatService(project_id="proj", db_path=db)
        assert s1.get_session_id() != s2.get_session_id()
