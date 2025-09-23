"""Document ingestion: a polite breadth-first web crawler plus loaders for local
files. Both produce LangChain `Document` objects carrying rich metadata
(`source`, `title`, `file_type`, `depth`) so downstream chunking, retrieval, and
source attribution all have something to point back at."""
from __future__ import annotations
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, List, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document

_BOILERPLATE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript")


def _extract(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return title, text


def _links(base_url: str, html: str) -> Iterator[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0]
        if href:
            yield urljoin(base_url, href)


def crawl_web(seeds: Iterable[str], max_pages: int = 5000, max_depth: int = 3,
              same_host_only: bool = True, delay_s: float = 0.1,
              timeout_s: float = 10.0) -> List[Document]:
    """BFS crawl from `seeds`, staying on-host by default, capped by page/depth.
    Returns one Document per fetched HTML page."""
    seeds = list(seeds)
    seen: Set[str] = set()
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    docs: List[Document] = []

    with httpx.Client(follow_redirects=True, timeout=timeout_s,
                      headers={"User-Agent": "SageCrawler/1.0"}) as http:
        while queue and len(docs) < max_pages:
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            try:
                resp = http.get(url)
                resp.raise_for_status()
            except httpx.HTTPError:
                continue
            if "text/html" not in resp.headers.get("content-type", ""):
                continue

            title, text = _extract(resp.text)
            docs.append(Document(page_content=text,
                                 metadata={"source": url, "title": title,
                                           "file_type": "html", "depth": depth}))
            if depth < max_depth:
                for link in _links(url, resp.text):
                    if link not in seen and (not same_host_only
                                             or urlparse(link).netloc == urlparse(seeds[0]).netloc):
                        queue.append((link, depth + 1))
            time.sleep(delay_s)
    return docs


def load_directory(path: str, glob: str = "**/*") -> List[Document]:
    """Load .html/.htm/.txt/.md/.rst files from a local tree as Documents."""
    root = Path(path)
    docs: List[Document] = []
    for fp in sorted(root.glob(glob)):
        if fp.suffix.lower() not in {".html", ".htm", ".txt", ".md", ".rst"}:
            continue
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        if fp.suffix.lower() in {".html", ".htm"}:
            title, text = _extract(raw)
        else:
            title, text = fp.stem, raw
        docs.append(Document(page_content=text,
                             metadata={"source": str(fp), "title": title,
                                       "file_type": fp.suffix.lstrip(".")}))
    return docs
