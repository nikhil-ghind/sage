"""Cleaning filters applied between ingestion and chunking: collapse whitespace,
drop navigation/boilerplate noise, enforce a minimum length, keep only allowed
languages, and remove exact + near-duplicate pages. Keeps the index small and the
retrieved context clean."""
from __future__ import annotations
import hashlib
import re
from typing import Iterable, List, Optional, Set

from langchain_core.documents import Document
from langdetect import LangDetectException, detect

_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Lines that are almost always navigation / chrome rather than content.
_NOISE_RE = re.compile(r"^(home|menu|search|login|sign in|cookie|subscribe|share)\b",
                       re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalise whitespace and drop obvious navigation lines."""
    lines = []
    for line in text.splitlines():
        line = _WS_RE.sub(" ", line).strip()
        if line and not _NOISE_RE.match(line):
            lines.append(line)
    return _BLANKLINES_RE.sub("\n\n", "\n".join(lines)).strip()


def detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None


def _shingles(text: str, k: int = 5) -> Set[int]:
    words = _WORD_RE.findall(text.lower())
    return {hash(" ".join(words[i:i + k])) for i in range(max(len(words) - k + 1, 1))}


def _jaccard(a: Set[int], b: Set[int]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def filter_documents(docs: Iterable[Document], min_words: int = 20,
                     allowed_languages: Optional[Iterable[str]] = None,
                     near_dup_threshold: float = 0.9) -> List[Document]:
    """Clean each document, then drop short, off-language, and duplicate pages.
    Tags each surviving doc with `metadata['lang']`."""
    allow = set(allowed_languages) if allowed_languages else None
    seen_hashes: Set[str] = set()
    seen_shingles: List[Set[int]] = []
    out: List[Document] = []

    for doc in docs:
        text = clean_text(doc.page_content)
        if len(_WORD_RE.findall(text)) < min_words:
            continue
        lang = detect_language(text)
        if allow is not None and lang not in allow:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        shingles = _shingles(text)
        if any(_jaccard(shingles, prev) >= near_dup_threshold for prev in seen_shingles):
            continue

        seen_hashes.add(digest)
        seen_shingles.append(shingles)
        meta = dict(doc.metadata)
        meta["lang"] = lang
        out.append(Document(page_content=text, metadata=meta))
    return out
