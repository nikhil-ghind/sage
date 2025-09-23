"""End-to-end ingestion: crawl / load → clean & filter → recursive chunk →
embed → upsert to Pinecone. Shared by the API and any CLI driver."""
from __future__ import annotations
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from src.embeddings.embedder import get_embeddings
from src.ingestion.chunker import chunk_documents
from src.ingestion.crawler import crawl_web, load_directory
from src.ingestion.filters import filter_documents
from src.vectorstore.pinecone_store import build_vectorstore


def ingest_documents(docs: List[Document], index_name: str,
                     embedding_provider: str = "openai",
                     embedding_model: str = "text-embedding-3-small",
                     chunk_strategy: str = "recursive", chunk_size: int = 512,
                     chunk_overlap: int = 64, min_words: int = 20,
                     allowed_languages: Optional[Iterable[str]] = None) -> int:
    """Filter → chunk → embed → upsert. Returns the number of chunks indexed."""
    clean = filter_documents(docs, min_words=min_words, allowed_languages=allowed_languages)
    chunks = chunk_documents(clean, chunk_strategy, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap)
    if not chunks:
        return 0
    embeddings = get_embeddings(embedding_provider, embedding_model)
    build_vectorstore(chunks, embeddings, index_name, model=embedding_model)
    return len(chunks)


def ingest_web(seeds: List[str], index_name: str, max_pages: int = 5000, **kwargs) -> int:
    return ingest_documents(crawl_web(seeds, max_pages=max_pages), index_name, **kwargs)


def ingest_directory(path: str, index_name: str, **kwargs) -> int:
    return ingest_documents(load_directory(path), index_name, **kwargs)
