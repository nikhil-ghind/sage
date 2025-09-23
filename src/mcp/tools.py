"""MCP tool definitions Sage's agent calls during retrieve-and-reason:
`search_docs` (semantic search), `filter_by_metadata` (scoped search), and
`get_page` (fetch all chunks of one source in order). Implemented as LangChain
`@tool`s so they can be bound to the LLM for tool-calling."""
from __future__ import annotations
from typing import Optional

from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore

from src.retrieval.retriever import retrieve

_vectorstore: Optional[VectorStore] = None


def register_vectorstore(vs: VectorStore) -> None:
    global _vectorstore
    _vectorstore = vs


def _format(docs) -> str:
    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('source', '?')} #${d.metadata.get('chunk_index', '?')}]\n"
        f"{d.page_content}" for d in docs)


@tool
def search_docs(query: str, k: int = 6) -> str:
    """Semantic search over the documentation corpus; returns the top-k passages."""
    return _format(retrieve(_vectorstore, query, k=k))


@tool
def filter_by_metadata(query: str, file_type: str = "", lang: str = "", k: int = 6) -> str:
    """Semantic search scoped by metadata (file_type and/or language)."""
    flt = {}
    if file_type:
        flt["file_type"] = file_type
    if lang:
        flt["lang"] = lang
    return _format(retrieve(_vectorstore, query, k=k, metadata_filter=flt or None))


@tool
def get_page(source: str, k: int = 20) -> str:
    """Fetch the chunks belonging to a specific source URL/path, in order."""
    docs = retrieve(_vectorstore, source, k=k, metadata_filter={"source": source})
    docs = sorted(docs, key=lambda d: d.metadata.get("chunk_index", 0))
    return "\n\n".join(d.page_content for d in docs)


MCP_TOOLS = [search_docs, filter_by_metadata, get_page]
