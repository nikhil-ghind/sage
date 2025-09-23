"""Pinecone serverless vector storage. Creates the index on demand and exposes
build / load helpers. Chunk metadata (source, title, file_type, lang,
chunk_index) is upserted alongside each vector so retrieval can apply metadata
filters and answers can cite their sources."""
from __future__ import annotations
import os
from typing import List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.embeddings.embedder import embedding_dimension


def init_pinecone(index_name: str, dimension: int = 1536, metric: str = "cosine",
                  cloud: str = "aws", region: str = "us-east-1") -> Pinecone:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    return pc


def build_vectorstore(docs: List[Document], embeddings, index_name: str,
                      model: str = "text-embedding-3-small",
                      namespace: str = "default") -> PineconeVectorStore:
    init_pinecone(index_name, dimension=embedding_dimension(model))
    return PineconeVectorStore.from_documents(
        docs, embeddings, index_name=index_name, namespace=namespace)


def load_vectorstore(embeddings, index_name: str,
                     namespace: str = "default") -> PineconeVectorStore:
    init_pinecone(index_name)
    return PineconeVectorStore(
        index_name=index_name, embedding=embeddings, namespace=namespace)
