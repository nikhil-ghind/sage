"""Recursive chunking. Splits documents on a hierarchy of separators
(paragraph → line → sentence → word) into size-bounded windows with overlap, so
each chunk is a coherent retrieval unit that keeps surrounding context. Metadata
(including `source` and `title`) is propagated and a `chunk_index` is added."""
from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


def chunk_recursive(docs: List[Document], chunk_size: int = 512,
                    chunk_overlap: int = 64) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _tag(splitter.split_documents(docs))


def chunk_by_tokens(docs: List[Document], max_tokens: int = 256,
                    chunk_overlap: int = 32) -> List[Document]:
    splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=chunk_overlap)
    return _tag(splitter.split_documents(docs))


def _tag(chunks: List[Document]) -> List[Document]:
    """Stamp a per-source chunk_index so metadata filters / citations can locate chunks."""
    counters: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        idx = counters.get(src, 0)
        c.metadata["chunk_index"] = idx
        counters[src] = idx + 1
    return chunks


def chunk_documents(docs: List[Document], strategy: str = "recursive",
                    **kwargs) -> List[Document]:
    if strategy == "recursive":
        return chunk_recursive(docs, **kwargs)
    if strategy == "token":
        return chunk_by_tokens(docs, **kwargs)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
