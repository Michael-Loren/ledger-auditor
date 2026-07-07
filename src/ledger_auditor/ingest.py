"""Ingestion: turn heterogeneous source files into Chunks (for retrieval)
and Transactions (for the structured query tool).

Real-world note: this is the part most RAG tutorials skip. PDF text comes out
messy — headings glued to body text, hyphenation, table cells out of order —
so ingestion is heading-aware rather than fixed-window."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pdfplumber

from .models import Chunk, Transaction

# Headings look like "3. Rent" or "Coverage A — Personal Property" or
# all-title-case short lines. Tuned for contract-style documents.
_HEADING_RE = re.compile(
    r"^(\d{1,2}\.\s+[A-Z][^\n]{2,60}|Coverage [A-Z][^\n]{0,60}|"
    r"Policy Information|Premium|Exclusions)\s*$"
)


def extract_pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def chunk_document(doc_id: str, raw_text: str) -> list[Chunk]:
    """Heading-aware chunking: one chunk per section, so a clause is never
    split across chunks (splitting clauses is the #1 cause of bad contract RAG)."""
    lines = [ln.strip() for ln in raw_text.splitlines()]
    chunks: list[Chunk] = []
    section = "preamble"
    buf: list[str] = []

    def flush():
        text = " ".join(buf).strip()
        if text:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:{len(chunks)}",
                doc_id=doc_id, section=section, text=f"[{section}] {text}"))

    for ln in lines:
        if _HEADING_RE.match(ln):
            flush()
            buf = []
            section = ln
        elif ln:
            buf.append(ln)
    flush()
    return chunks


def ingest_docs(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        doc_id = pdf_path.stem.replace("_policy", "")  # lease.pdf -> lease
        chunks.extend(chunk_document(doc_id, extract_pdf_text(pdf_path)))
    return chunks


# ------------------------------------------------------------------ statements
def _parse_csv_statement(path: Path) -> list[Transaction]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(Transaction(
                date=row["date"], description=row["description"].strip(),
                amount=float(row["amount"]), category=row.get("category", ""),
                source_file=path.name))
    return out


def _parse_pdf_statement(path: Path) -> list[Transaction]:
    """Extract the transaction table from a PDF statement."""
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row or row[0] in (None, "Date"):
                        continue
                    date, desc, amount = row[0], row[1], row[2]
                    if not re.match(r"\d{4}-\d{2}-\d{2}", str(date or "")):
                        continue
                    out.append(Transaction(
                        date=str(date), description=str(desc).strip(),
                        amount=float(str(amount).replace(",", "")),
                        source_file=path.name))
    return out


def ingest_statements(statements_dir: Path, prefer_csv: bool = True) -> list[Transaction]:
    """Load all statements. When a month exists as both CSV and PDF, prefer one
    (default CSV) and use the other for parser cross-validation in tests."""
    txns: list[Transaction] = []
    csv_months = {p.stem for p in statements_dir.glob("*.csv")}
    for p in sorted(statements_dir.iterdir()):
        if p.suffix == ".csv":
            txns.extend(_parse_csv_statement(p))
        elif p.suffix == ".pdf" and (p.stem not in csv_months or not prefer_csv):
            txns.extend(_parse_pdf_statement(p))
    txns.sort(key=lambda t: t.date)
    return txns
