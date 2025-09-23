"""LangFuse tracing + RAG evaluation used for quality tracking and prompt
optimization: trace every agent run, and score retrieval recall, faithfulness,
and answer relevance so prompt/retrieval changes can be compared on the same set."""
from __future__ import annotations
import os
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

from src.agents.rag_agent import run_query

_langfuse: Optional[Langfuse] = None


def init_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _langfuse


@observe(name="sage_rag_query")
def traced_query(agent, question: str) -> str:
    answer = run_query(agent, question)
    langfuse_context.update_current_trace(input=question, output=answer.text)
    return answer.text


def retrieval_recall(retrieved_sources: List[str], gold_sources: List[str]) -> float:
    if not gold_sources:
        return 0.0
    return len(set(retrieved_sources) & set(gold_sources)) / len(set(gold_sources))


def _judge(system: str, payload: Dict, model: str = "gpt-4o-mini") -> float:
    llm = ChatOpenAI(model=model, temperature=0)
    keys = ", ".join(f"{{{k}}}" for k in payload)
    result = (ChatPromptTemplate.from_messages(
        [("system", system), ("human", keys)]) | llm).invoke(payload)
    try:
        return min(1.0, max(0.0, float(result.content.strip().split()[0])))
    except (ValueError, IndexError):
        return 0.0


def score_faithfulness(context: str, answer: str) -> float:
    return _judge("Score 0-1 how fully the ANSWER is supported by CONTEXT. "
                  "Reply with only a float.", {"context": context, "answer": answer})


def score_relevance(question: str, answer: str) -> float:
    return _judge("Score 0-1 how well the ANSWER addresses the QUESTION. "
                  "Reply with only a float.", {"question": question, "answer": answer})


def evaluate_batch(qa_pairs: List[Dict], agent, lf: Optional[Langfuse] = None) -> List[Dict]:
    lf = lf or init_langfuse()
    results = []
    for pair in qa_pairs:
        q = pair["question"]
        answer = run_query(agent, q)
        context = "\n".join(s["source"] for s in answer.sources)
        faith = score_faithfulness(context, answer.text)
        rel = score_relevance(q, answer.text)
        results.append({"question": q, "faithfulness": faith, "relevance": rel,
                        "iterations": answer.iterations})
        lf.score(name="faithfulness", value=faith, comment=q)
        lf.score(name="relevance", value=rel, comment=q)
    return results
