"""Retrieval over the Pinecone store. Supports MMR (diversity-aware) search and
Pinecone metadata filtering, so the agent can scope a query to e.g. a file_type,
language, or source section."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def build_retriever(vectorstore: VectorStore, search_type: str = "mmr",
                    k: int = 6, fetch_k: int = 24,
                    metadata_filter: Optional[Dict] = None):
    kwargs: Dict = {"k": k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    if search_type == "mmr":
        kwargs["fetch_k"] = fetch_k
        return vectorstore.as_retriever(search_type="mmr", search_kwargs=kwargs)
    return vectorstore.as_retriever(search_kwargs=kwargs)


def retrieve(vectorstore: VectorStore, query: str, k: int = 6,
             metadata_filter: Optional[Dict] = None) -> List[Document]:
    return build_retriever(vectorstore, k=k, metadata_filter=metadata_filter).invoke(query)


def retrieve_with_scores(vectorstore: VectorStore, query: str, k: int = 6,
                         metadata_filter: Optional[Dict] = None
                         ) -> List[Tuple[Document, float]]:
    """Similarity search returning (doc, score). Used by the agent's confidence
    check to decide whether the retrieved context is strong enough to answer."""
    return vectorstore.similarity_search_with_relevance_scores(
        query, k=k, filter=metadata_filter)
