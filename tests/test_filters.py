from langchain_core.documents import Document

from src.ingestion.filters import clean_text, filter_documents


def test_clean_collapses_whitespace_and_drops_nav():
    raw = "Home\nSearch\nThe   actual    content   line.\n\n\n\nMore content here."
    cleaned = clean_text(raw)
    assert "Home" not in cleaned.splitlines()
    assert "The actual content line." in cleaned
    assert "\n\n\n" not in cleaned


def test_drops_short_documents():
    docs = [Document(page_content="too short", metadata={"source": "a"}),
            Document(page_content="word " * 50, metadata={"source": "b"})]
    kept = filter_documents(docs, min_words=20)
    assert [d.metadata["source"] for d in kept] == ["b"]


def test_exact_duplicate_removed():
    body = "Sage performs agentic retrieval over documentation pages with metadata filters."
    docs = [Document(page_content=body, metadata={"source": "a"}),
            Document(page_content=body, metadata={"source": "b"})]
    kept = filter_documents(docs, min_words=5)
    assert len(kept) == 1


def test_near_duplicate_removed():
    # A long, varied document and a copy with one extra sentence: shingle-Jaccard
    # stays above threshold, so the near-duplicate is dropped.
    base = " ".join(f"Section {i} explains configuration option number {i} in full detail."
                    for i in range(40))
    docs = [Document(page_content=base, metadata={"source": "a"}),
            Document(page_content=base + " One additional closing note here.",
                     metadata={"source": "b"})]
    kept = filter_documents(docs, min_words=10, near_dup_threshold=0.9)
    assert len(kept) == 1
    assert kept[0].metadata["source"] == "a"


def test_language_allowlist_and_tagging():
    en = Document(page_content="The quick brown fox jumps over the lazy dog by the river. " * 3,
                  metadata={"source": "en"})
    fr = Document(page_content="Le chat noir dort sur le canapé du salon toute la journée. " * 3,
                  metadata={"source": "fr"})
    kept = filter_documents([en, fr], min_words=10, allowed_languages=["en"])
    assert len(kept) == 1
    assert kept[0].metadata["lang"] == "en"
