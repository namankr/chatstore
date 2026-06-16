"""
chatstore
~~~~~~~~~
Lightweight persistent chat service with optional semantic search.

v1 — basic usage::

    from chatstore import ChatService

    chat = ChatService(project_id="my_app")
    chat.save_message("user", "Hello!")
    chat.save_message("assistant", "Hi there!")
    print(chat.load_history())

v2 — with semantic search (requires ``pip install chatstore[semantic]``)::

    chat = ChatService(
        project_id="my_app",
        enable_semantic_search=True,
    )
    chat.store_in_memory("Some tool result text", source="web_search")
    context = chat.retrieve_from_memory("my query")
"""

from chatstore.service import ChatService

__all__ = ["ChatService"]
__version__ = "2.0.0"
