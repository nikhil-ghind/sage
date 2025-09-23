"""FastAPI surface for Sage: ingest a corpus, query it through the agentic RAG
graph, and report health. The vector store + agent are wired once at startup;
ingestion runs in the background."""
from __future__ import annotations
import os
from typing import Dict, List, Optional

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.rag_agent import build_rag_agent, run_query
from src.embeddings.embedder import get_embeddings
from src.ingestion.pipeline import ingest_directory, ingest_web
from src.mcp.tools import register_vectorstore
from src.vectorstore.pinecone_store import load_vectorstore

app = FastAPI(title="Sage — Agentic RAG & Semantic Search")

_cfg: dict = {}
_agent = None


def _load_config() -> dict:
    with open(os.getenv("CONFIG_PATH", "configs/config.yaml")) as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def startup() -> None:
    global _cfg, _agent
    _cfg = _load_config()
    embeddings = get_embeddings(_cfg.get("embedding_provider", "openai"),
                                _cfg.get("embedding_model"))
    vs = load_vectorstore(embeddings, _cfg["pinecone_index"])
    register_vectorstore(vs)
    _agent = build_rag_agent(vs, model=_cfg.get("llm_model", "gpt-4o-mini"),
                             k=_cfg.get("retriever_k", 6),
                             confidence_threshold=_cfg.get("confidence_threshold", 0.7),
                             max_iterations=_cfg.get("max_iterations", 3))


class QueryRequest(BaseModel):
    question: str
    metadata_filter: Optional[Dict] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    query_type: str
    iterations: int


class IngestRequest(BaseModel):
    seeds: Optional[List[str]] = None
    directory: Optional[str] = None
    max_pages: int = 5000
    allowed_languages: Optional[List[str]] = None


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if _agent is None:
        raise HTTPException(503, "Agent not initialised")
    a = run_query(_agent, req.question, metadata_filter=req.metadata_filter)
    return QueryResponse(question=req.question, answer=a.text, sources=a.sources,
                         query_type=a.query_type, iterations=a.iterations)


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    if not req.seeds and not req.directory:
        raise HTTPException(400, "Provide either 'seeds' or 'directory'")
    index = _cfg["pinecone_index"]

    def _run() -> None:
        if req.directory:
            ingest_directory(req.directory, index, allowed_languages=req.allowed_languages)
        else:
            ingest_web(req.seeds, index, max_pages=req.max_pages,
                       allowed_languages=req.allowed_languages)

    background_tasks.add_task(_run)
    return {"status": "ingestion started"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "agent_ready": _agent is not None}
