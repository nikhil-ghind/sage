"""Embedding generation. Defaults to OpenAI `text-embedding-3-small` (1536-dim),
with a HuggingFace fallback for offline / cost-sensitive runs."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embeddings(provider: str = "openai", model: str | None = None):
    if provider == "openai":
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")
    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedding provider: {provider}")


def embedding_dimension(model: str = "text-embedding-3-small") -> int:
    return DIMENSIONS.get(model, 1536)
