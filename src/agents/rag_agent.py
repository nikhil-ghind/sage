"""Agentic RAG on LangGraph.

Graph:  router → retrieve → reflect → (rewrite → retrieve …) | generate → END

- router   classifies the query (simple_lookup / multi_hop / metadata_scoped) and
           sets an optional metadata filter.
- retrieve pulls scored context from Pinecone (honouring the router's filter).
- reflect  is the self-reflection step: if the retrieved context is low-confidence
           (best relevance score below threshold) and retries remain, the query is
           rewritten and retrieval runs again; otherwise we proceed to generate.
- generate produces a grounded answer with inline [n] source attribution.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.retrieval.retriever import retrieve_with_scores

ROUTER_PROMPT = (
    "Classify the user question into exactly one of: simple_lookup, multi_hop, "
    "metadata_scoped. Reply with only the label.")

GENERATE_PROMPT = """You are Sage, a documentation assistant. Answer the question \
using ONLY the numbered context passages. Cite the passages you use with inline \
markers like [1], [2]. If the context is insufficient, say so plainly.

Context:
{context}

Question: {question}"""

REWRITE_PROMPT = (
    "The retrieved context was weak. Rewrite the question into a more specific, "
    "keyword-rich search query. Reply with only the rewritten query.")


@dataclass
class Answer:
    text: str
    sources: List[Dict] = field(default_factory=list)
    query_type: str = ""
    iterations: int = 0


class AgentState(TypedDict, total=False):
    query: str
    original_query: str
    query_type: str
    metadata_filter: Optional[Dict]
    context: List[Document]
    best_score: float
    iterations: int
    answer: Answer


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
        for i, d in enumerate(docs))


def build_rag_agent(vectorstore: VectorStore, model: str = "gpt-4o-mini",
                    k: int = 6, confidence_threshold: float = 0.7,
                    max_iterations: int = 3):
    llm = ChatOpenAI(model=model, temperature=0)

    def router_node(state: AgentState) -> dict:
        label = (ChatPromptTemplate.from_messages(
            [("system", ROUTER_PROMPT), ("human", "{q}")]) | llm | StrOutputParser()
        ).invoke({"q": state["query"]}).strip().lower()
        if label not in {"simple_lookup", "multi_hop", "metadata_scoped"}:
            label = "simple_lookup"
        return {"query_type": label, "iterations": 0}

    def retrieve_node(state: AgentState) -> dict:
        scored = retrieve_with_scores(vectorstore, state["query"], k=k,
                                      metadata_filter=state.get("metadata_filter"))
        docs = [d for d, _ in scored]
        best = max((s for _, s in scored), default=0.0)
        return {"context": docs, "best_score": best,
                "iterations": state.get("iterations", 0) + 1}

    def rewrite_node(state: AgentState) -> dict:
        new_q = (ChatPromptTemplate.from_messages(
            [("system", REWRITE_PROMPT), ("human", "{q}")]) | llm | StrOutputParser()
        ).invoke({"q": state["original_query"]}).strip()
        return {"query": new_q or state["query"]}

    def generate_node(state: AgentState) -> dict:
        docs = state["context"]
        text = (ChatPromptTemplate.from_messages([("system", GENERATE_PROMPT)]) | llm
                | StrOutputParser()).invoke(
            {"context": _format_context(docs), "question": state["original_query"]})
        sources = [{"n": i + 1, "source": d.metadata.get("source", "?"),
                    "title": d.metadata.get("title", "")} for i, d in enumerate(docs)]
        return {"answer": Answer(text=text, sources=sources,
                                 query_type=state.get("query_type", ""),
                                 iterations=state.get("iterations", 0))}

    def reflect(state: AgentState) -> str:
        confident = state.get("best_score", 0.0) >= confidence_threshold
        if confident or state.get("iterations", 0) >= max_iterations:
            return "generate"
        return "rewrite"

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "retrieve")
    graph.add_conditional_edges("retrieve", reflect,
                                {"rewrite": "rewrite", "generate": "generate"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def run_query(agent, question: str, metadata_filter: Optional[Dict] = None) -> Answer:
    state = agent.invoke({
        "query": question,
        "original_query": question,
        "metadata_filter": metadata_filter,
        "iterations": 0,
    })
    return state["answer"]
