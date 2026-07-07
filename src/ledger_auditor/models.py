"""Shared data models."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    """A retrievable unit of document text."""
    chunk_id: str
    doc_id: str          # e.g. "lease", "insurance"
    section: str         # heading the chunk belongs to
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transaction:
    date: str            # ISO yyyy-mm-dd
    description: str
    amount: float        # negative = debit
    category: str = ""
    source_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class Citation:
    doc_id: str
    section: str
    quote: str


@dataclass
class AuditAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    verified: bool = False
    verification_notes: str = ""
