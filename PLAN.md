# Sage

## Project Overview

Sage is an agentic RAG and semantic-search system over large documentation
corpora (10k+ pages). A Python pipeline crawls and cleans documents, chunks them
recursively, embeds them with OpenAI, and stores them in Pinecone with metadata.
At query time a LangGraph agent routes the question, retrieves and reasons over
context via MCP tools, and—crucially—self-reflects: when the retrieved context is
low-confidence it rewrites the query and retrieves again before generating a
grounded answer with source attribution. LangFuse traces and scores every run to
drive prompt optimization and latency reduction.

Key goals:
- Low-latency natural-language semantic search across 10k+ pages
- High-relevance retrieval via recursive chunking, metadata filtering, and MMR
- Agentic multi-hop reasoning with self-reflection re-query on weak context
- Grounded answers that attribute their sources
- Full observability of LLM/retrieval quality via LangFuse

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Orchestration | LangChain + LangGraph |
| Tool protocol | MCP (LangChain `@tool`) |
| Embeddings | OpenAI `text-embedding-3-small` (HuggingFace fallback) |
| LLM | OpenAI `gpt-4o-mini` / `gpt-4o` |
| Vector DB | Pinecone (serverless, metadata filtering) |
| Crawl / parse | httpx + BeautifulSoup |
| Cleaning | langdetect, shingle-Jaccard dedup |
| Observability | LangFuse |
| Testing | pytest |

---

## Architecture Overview

```
Ingestion pipeline
  crawl_web / load_directory → filter_documents → chunk_documents
      → OpenAI embeddings → Pinecone upsert (vectors + metadata)

Query-time (LangGraph AgentState)
  router_node:   classify (simple_lookup | multi_hop | metadata_scoped)
  retrieve_node: MMR + metadata filter, scored
  reflect:       best_score ≥ threshold or iters ≥ max → generate
                 else → rewrite_node → retrieve   (self-reflection re-query)
  generate_node: grounded answer with inline [n] source attribution

Observability
  LangFuse @observe traces; scores recall + faithfulness + relevance
```

---

## Phase 1 — Scaffolding & Config

- Directory layout under `src/` (ingestion, embeddings, vectorstore, retrieval,
  mcp, agents, evaluation), `api/`, `configs/`, `tests/`.
- `requirements.txt`, `.env.example`, `configs/config.yaml` (index name, models,
  chunking, `retriever_k`, `confidence_threshold`, `max_iterations`).

## Phase 2 — Ingestion: Crawl, Clean, Chunk

- `crawler.py`: BFS `crawl_web` (host-scoped, depth/page caps, polite delay,
  boilerplate stripping) + `load_directory` for local html/txt/md/rst → Documents.
- `filters.py`: `clean_text` (whitespace + nav removal), `detect_language`,
  min-word gate, exact (sha256) + near-dup (shingle-Jaccard) removal; tags `lang`.
- `chunker.py`: recursive + token splitting with overlap; per-source `chunk_index`.
- `pipeline.py`: crawl/load → filter → chunk → embed → upsert.
- Tests: `test_chunker.py`, `test_filters.py`.

## Phase 3 — Embeddings & Pinecone

- `embedder.py`: OpenAI embeddings + dimension lookup; HuggingFace fallback.
- `pinecone_store.py`: serverless index init (dim from model), `build_vectorstore`
  / `load_vectorstore`; chunk metadata upserted for filtered retrieval.

## Phase 4 — Retrieval & MCP Tools

- `retriever.py`: `build_retriever` (MMR + metadata filter), `retrieve`, and
  `retrieve_with_scores` (relevance scores feed the confidence check).
- `mcp/tools.py`: `search_docs`, `filter_by_metadata`, `get_page` as `@tool`s
  bound to a shared vector store.

## Phase 5 — Agentic Graph

- `agents/rag_agent.py`: LangGraph with `router`, `retrieve`, `rewrite`,
  `generate` nodes and a `reflect` conditional edge implementing the
  self-reflection re-query loop bounded by `max_iterations`. `run_query` returns
  an `Answer` with text + structured sources + query_type + iterations.

## Phase 6 — API, Evaluation & Tuning

- `api/main.py`: `/ingest` (background), `/query` (optional metadata filter),
  `/health`; agent + store wired at startup.
- `langfuse_eval.py`: `@observe` tracing, `retrieval_recall`, faithfulness /
  relevance judges, `evaluate_batch` for prompt optimization and latency tuning.
- Smoke test:
  ```bash
  curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
    -d '{"question": "How do I rotate API keys?"}'
  ```
