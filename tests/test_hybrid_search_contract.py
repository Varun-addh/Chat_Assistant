"""Contract tests for HybridSearchEngine.hybrid_search().

Regression guard. The consumer in interview_intelligence_service merges hybrid
scores back into its own result list by matching on the ``question`` key and
reading ``hybrid_score``::

    for result in all_results:
        for hybrid in hybrid_results:
            if result.get('question') == hybrid.get('question'):
                result['hybrid_score'] = hybrid.get('hybrid_score', 0.5)

A previous change reshaped the return value to {content, metadata, score, rank}
to suit failure capture. Both keys the consumer reads then vanished, so the
match never fired, every score silently defaulted to 0.5, and BM25 + semantic
fusion became a no-op with no exception and no log line.

These tests pin the keys the consumer depends on, and exercise the merge itself
so the coupling breaks loudly instead of silently.
"""

import pytest

from app.services.chat.ai_native_enhancements import HybridSearchEngine


class _FakeDoc:
    def __init__(self, question, source="github", credibility=0.9):
        self.metadata = {
            "question": question,
            "source": source,
            "credibility": credibility,
        }


class _FakeBM25:
    def __init__(self, questions):
        self._docs = [_FakeDoc(q) for q in questions]

    def invoke(self, query):
        return list(self._docs)


class _FakeHit:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class _FakeQdrant:
    def __init__(self, hits):
        self._hits = hits

    def search(self, collection_name, query_vector, limit, score_threshold):
        return list(self._hits)


class _FakeEmbeddings:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


def _engine(bm25_questions=(), semantic_hits=()):
    """Build an engine without touching __init__ (which needs a real Qdrant)."""
    engine = HybridSearchEngine.__new__(HybridSearchEngine)
    engine.bm25_retriever = _FakeBM25(list(bm25_questions)) if bm25_questions else None
    engine.qdrant_client = _FakeQdrant(list(semantic_hits))
    engine.embeddings = _FakeEmbeddings()
    engine.collection_name = "test_collection"
    return engine


@pytest.mark.asyncio
async def test_results_expose_question_and_hybrid_score():
    """The two keys the consumer reads must be present at the top level."""
    engine = _engine(
        bm25_questions=["What is a goroutine?"],
        semantic_hits=[_FakeHit({"question": "Explain channels", "answer": "..."}, 0.8)],
    )

    results = await engine.hybrid_search(query="go concurrency", k=10)

    assert results, "expected hybrid results"
    for r in results:
        assert "question" in r, f"consumer matches on 'question'; got keys {sorted(r)}"
        assert "hybrid_score" in r, f"consumer reads 'hybrid_score'; got keys {sorted(r)}"
        assert isinstance(r["hybrid_score"], (int, float))
        assert r["retrieval_method"] == "hybrid"


@pytest.mark.asyncio
async def test_consumer_merge_actually_assigns_scores():
    """Replicates the merge loop in interview_intelligence_service."""
    engine = _engine(
        bm25_questions=["Q1", "Q2"],
        semantic_hits=[_FakeHit({"question": "Q3"}, 0.7)],
    )

    hybrid_results = await engine.hybrid_search(query="q", k=10)

    all_results = [{"question": "Q1"}, {"question": "Q2"}, {"question": "Q3"}]
    for result in all_results:
        for hybrid in hybrid_results:
            if result.get("question") == hybrid.get("question"):
                result["hybrid_score"] = hybrid.get("hybrid_score", 0.5)
                break

    assigned = [r for r in all_results if "hybrid_score" in r]
    assert len(assigned) == 3, "every result should receive a real hybrid score"
    # The bug produced exactly this: nothing matched, so nothing was assigned,
    # and the later sort fell back to a flat default.
    assert not all(r["hybrid_score"] == 0.5 for r in assigned), (
        "all scores collapsed to the 0.5 default — the merge did not match"
    )


@pytest.mark.asyncio
async def test_semantic_only_results_preserve_payload_fields():
    """Semantic hits are spread, so payload keys stay reachable to the consumer."""
    engine = _engine(
        semantic_hits=[
            _FakeHit({"question": "Explain B-trees", "answer": "A B-tree is...", "topic": "DB"}, 0.9)
        ],
    )

    results = await engine.hybrid_search(query="indexing", k=5)

    assert len(results) == 1
    assert results[0]["question"] == "Explain B-trees"
    assert results[0]["answer"] == "A B-tree is..."
    assert results[0]["topic"] == "DB"
    assert results[0]["hybrid_score"] > 0


@pytest.mark.asyncio
async def test_ranking_orders_by_combined_score():
    """A question found by both retrievers outranks one found by only one."""
    engine = _engine(
        bm25_questions=["Both", "KeywordOnly"],
        semantic_hits=[_FakeHit({"question": "Both"}, 0.9)],
    )

    results = await engine.hybrid_search(query="q", k=10)

    assert results[0]["question"] == "Both"
    scores = [r["hybrid_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_empty_retrievers_return_empty_not_error():
    engine = _engine()
    assert await engine.hybrid_search(query="nothing", k=5) == []
