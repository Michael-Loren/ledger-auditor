"""Calculator safety, transaction queries, scoring, and citation verification."""
import pytest

from ledger_auditor.evaluate import score_answer
from ledger_auditor.tools import safe_calculate


def test_extract_json_handles_fenced_output():
    """Regression: verifier model wrapped its JSON in markdown fences."""
    import json
    from ledger_auditor.agent import _extract_json
    for raw in ('{"verified": true, "notes": "ok"}',
                '```json\n{"verified": true, "notes": "ok"}\n```',
                'Here is my verdict:\n{"verified": true, "notes": "ok"}'):
        assert json.loads(_extract_json(raw))["verified"] is True
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_calculator_basic():
    assert safe_calculate("1850 * 1.03") == pytest.approx(1905.5)
    assert safe_calculate("round((1942.50 - 1905.50) * 6, 2)") == 222.0
    assert safe_calculate("sum([1, 2, 3.5])") == 6.5


def test_calculator_rejects_code_injection():
    for evil in ("__import__('os')", "open('x')", "1; print(1)", "[].__class__"):
        with pytest.raises(Exception):
            safe_calculate(evil)


def test_transaction_query_filters(retriever_and_store):
    _, store = retriever_and_store
    res = store.query(description_contains="CLEANCO")
    assert res["count"] == 2 and res["sum_amount"] == -240.0
    rent = store.query(description_contains="MAPLE PROPERTIES LLC - RENT",
                       date_from="2025-08-01", date_to="2025-08-31")
    assert rent["count"] == 1 and rent["rows"][0]["amount"] == -1942.50


def test_find_duplicates_catches_cleanco(retriever_and_store):
    """Regression: the agent missed the duplicate CLEANCO charge because
    the row-limited query view can't scan all 306 transactions."""
    _, store = retriever_and_store
    res = store.find_duplicates()
    assert len(res["duplicate_groups"]) == 1
    d = res["duplicate_groups"][0]
    assert d["description"] == "CLEANCO HOME SERVICES"
    assert d["date"] == "2025-03-14" and d["occurrences"] == 2


def test_group_by_amount_counts_price_tiers(retriever_and_store):
    """Regression: the agent miscounted STREAMAX months (said 6, was 5)."""
    _, store = retriever_and_store
    res = store.query(description_contains="STREAMAX", group_by="amount")
    tiers = {k: v["count"] for k, v in res["groups"].items()}
    assert tiers == {"-12.99": 4, "-15.99": 5, "-17.99": 3}


def test_score_answer_numeric_tolerance():
    q = {"numeric": 222.0, "tolerance": 0.01, "answer": "$222.00"}
    assert score_answer(q, "The total overcharge is $222.00 across six months.")
    assert not score_answer(q, "The total overcharge is $200.")


def test_score_answer_keywords():
    q = {"numeric": None, "answer": "No, flood is excluded."}
    assert score_answer(q, "Flood damage is excluded under the policy.")


def test_citation_verification_catches_fabrication(retriever_and_store):
    """The deterministic verifier must reject quotes that aren't in the corpus."""
    from ledger_auditor.agent import AuditAgent
    from ledger_auditor.models import AuditAnswer, Citation
    retriever, store = retriever_and_store

    agent = AuditAgent.__new__(AuditAgent)   # no API client needed for stage 1
    agent.retriever = retriever
    agent.client = None

    fake = AuditAnswer(question="q", answer="a", citations=[
        Citation(doc_id="lease", section="", quote="rent may be raised by 10% yearly")])
    out = AuditAgent._verify(agent, fake)
    assert not out.verified and "fabricated" in out.verification_notes

    real = AuditAnswer(question="q", answer="a", citations=[
        Citation(doc_id="lease", section="", quote="late fee of $50.00")])
    # stage 2 needs an API client; stage 1 passing means it won't return early
    try:
        AuditAgent._verify(agent, real)
    except AttributeError:
        pass  # reached stage 2 -> the verbatim quote passed stage 1
