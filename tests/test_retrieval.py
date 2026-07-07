"""Retrieval quality gates — CI fails if hybrid recall regresses."""
from ledger_auditor.evaluate import load_questions, retrieval_eval


def test_search_finds_late_fee_clause(retriever_and_store):
    retriever, _ = retriever_and_store
    results = retriever.search("what late fee can my landlord charge?", k=3)
    assert any("late fee" in r.chunk.text.lower() and r.chunk.doc_id == "lease"
               for r in results)


def test_doc_filter(retriever_and_store):
    retriever, _ = retriever_and_store
    results = retriever.search("premium", k=3, doc_filter="insurance")
    assert all(r.chunk.doc_id == "insurance" for r in results)


def test_hybrid_recall_floor(retriever_and_store, data_dir):
    """Quality gate: hybrid recall@5 must stay >= 0.85 and beat-or-match
    each individual ranker (with the TF-IDF fallback backend)."""
    retriever, _ = retriever_and_store
    questions = load_questions(data_dir / "questions.jsonl")
    res = retrieval_eval(retriever, questions, k=5)
    r = res["recall_at_k"]
    assert r["hybrid"] >= 0.85, f"hybrid recall regressed: {r}"
    assert r["hybrid"] >= max(r["bm25"], r["dense"]) - 0.05, r
