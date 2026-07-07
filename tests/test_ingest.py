"""Ingestion tests, including CSV-vs-PDF parser cross-validation."""
from pathlib import Path

from ledger_auditor.ingest import (_parse_csv_statement, _parse_pdf_statement,
                                   ingest_docs, ingest_statements)


def test_lease_chunks_capture_key_clauses(data_dir):
    chunks = ingest_docs(data_dir / "docs")
    lease_text = " ".join(c.text.lower() for c in chunks if c.doc_id == "lease")
    for phrase in ("$1,850.00", "late fee of $50.00", "3% of the then-current",
                   "security deposit of $2,775.00", "pet fee", "60 days"):
        assert phrase.lower() in lease_text, f"missing clause text: {phrase}"


def test_insurance_chunks_capture_premium(data_dir):
    chunks = ingest_docs(data_dir / "docs")
    ins = " ".join(c.text.lower() for c in chunks if c.doc_id == "insurance")
    assert "$18.50" in ins and "deductible" in ins


def test_chunks_are_section_scoped(data_dir):
    chunks = ingest_docs(data_dir / "docs")
    sections = {c.section for c in chunks if c.doc_id == "lease"}
    assert any("Late Payments" in s for s in sections)
    assert any("Rent Increases" in s for s in sections)


def test_statement_count_and_range(data_dir):
    txns = ingest_statements(data_dir / "statements")
    assert len(txns) > 300
    assert txns[0].date.startswith("2025-01")
    assert txns[-1].date.startswith("2025-12")


def test_pdf_and_csv_parsers_agree(data_dir):
    """The same month parsed from PDF and CSV must produce identical totals."""
    sdir = data_dir / "statements"
    for month in ("2025-02", "2025-08"):
        csv_t = _parse_csv_statement(Path(sdir / f"{month}.csv"))
        pdf_t = _parse_pdf_statement(Path(sdir / f"{month}.pdf"))
        assert len(csv_t) == len(pdf_t)
        assert round(sum(t.amount for t in csv_t), 2) == \
               round(sum(t.amount for t in pdf_t), 2)
