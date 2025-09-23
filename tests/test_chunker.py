from langchain_core.documents import Document

from src.ingestion.chunker import chunk_recursive


def make_docs(n=1, length=2000):
    return [Document(page_content="word " * (length // 5),
                     metadata={"source": f"doc{i}", "title": f"Doc {i}"})
            for i in range(n)]


def test_recursive_chunks_smaller():
    chunks = chunk_recursive(make_docs(1, 2000), chunk_size=256, chunk_overlap=32)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_metadata_and_chunk_index_propagate():
    chunks = chunk_recursive(make_docs(1, 2000), chunk_size=256, chunk_overlap=32)
    for i, c in enumerate(chunks):
        assert c.metadata["source"] == "doc0"
        assert c.metadata["title"] == "Doc 0"
        assert c.metadata["chunk_index"] == i


def test_chunk_index_is_per_source():
    chunks = chunk_recursive(make_docs(2, 1500), chunk_size=256, chunk_overlap=32)
    for src in {"doc0", "doc1"}:
        idxs = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == src]
        assert idxs == list(range(len(idxs)))  # restarts at 0 per source
