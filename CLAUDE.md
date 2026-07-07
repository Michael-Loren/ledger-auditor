# CLAUDE.md — agent guide for ledger-auditor

This file orients an AI coding agent working in this repo. Read it before
editing anything.

## What this project is

An agentic RAG system that **audits** personal financial documents against
each other (lease + insurance policy vs. actual bank transactions), rather
than just answering questions about them. The demo dataset is fully synthetic
and deterministic; six anomalies are planted with known ground truth.

## Commands

```bash
pip install -e ".[dev]"                # dev install (TF-IDF fallback backend, no model download)
pip install -e ".[dense,dev]"          # + real sentence-transformers backend

python scripts/generate_data.py        # REQUIRED FIRST: creates data/ (gitignored)

pytest                                 # 14 tests; no API key or network needed
ledger-auditor ingest                  # corpus stats sanity check
ledger-auditor eval --mode retrieval   # retrieval benchmark, no API key needed
ledger-auditor eval --mode retrieval --k 3   # stricter k

# these need ANTHROPIC_API_KEY:
ledger-auditor ask "<question>" [-v]   # one question through the agent
ledger-auditor audit [-v]              # full anomaly sweep
ledger-auditor eval --mode full        # 50-question end-to-end eval (costs ~$1-2 in API calls)
```

Use `--backend tfidf` to force the dependency-light dense backend (this is
what CI uses; results differ from the sentence-transformers backend).

## Map of the code

| File | Responsibility | Key invariant |
|------|----------------|---------------|
| `scripts/generate_data.py` | Synthetic dataset + ground truth + the 50-question eval set | Eval answers are COMPUTED from the same constants that generate the data — never hand-edit `data/questions.jsonl`; change the generator |
| `src/ledger_auditor/models.py` | Dataclasses: Chunk, Transaction, Citation, AuditAnswer | — |
| `src/ledger_auditor/ingest.py` | PDF/CSV → Chunks + Transactions | Chunking is heading-aware: one contract section per chunk, a clause is never split |
| `src/ledger_auditor/retrieval.py` | BM25 + pluggable dense backend, RRF fusion | Dense backend must implement `fit(texts)` / `scores(query)`; `search()` supports mode=bm25/dense/hybrid so eval can benchmark each |
| `src/ledger_auditor/tools.py` | Agent tools: search_documents, query_transactions, calculate | Calculator is AST-whitelisted, NO eval(). Transactions are never embedded — they stay in `TransactionStore` and sums are computed in Python |
| `src/ledger_auditor/agent.py` | Tool-use loop + two-stage verifier | Stage 1 of `_verify` is deterministic (verbatim quote must exist in cited doc) and runs BEFORE any LLM judging. Do not weaken this to fuzzy matching |
| `src/ledger_auditor/evaluate.py` | Retrieval eval + end-to-end eval + answer scoring | `retrieval_eval` only uses questions tagged `retrieval_eval: true` (31 of 50) |
| `src/ledger_auditor/cli.py` | `ingest / ask / audit / eval` subcommands | Commands that need the API fail fast with a clear message if `ANTHROPIC_API_KEY` is unset |

## Architectural rules (do not break)

1. **The LLM never does arithmetic.** All math goes through the `calculate`
   tool or is computed by `TransactionStore.query` (count/sum). If you add a
   feature that needs math, add it to the store or calculator.
2. **Citations are verbatim or rejected.** `submit_answer` quotes must appear
   literally in the corpus (whitespace/case-normalized). The verifier's string
   check is a hard gate and must stay ahead of the LLM entailment check.
3. **Structured data stays structured.** Never embed transaction rows into
   the vector index. Documents → chunks; transactions → typed store.
4. **Chunking is section-scoped.** `_HEADING_RE` in ingest.py defines what a
   heading is. If you add a new document type whose headings don't match,
   extend the regex — don't fall back to fixed-window chunking.
5. **Everything is deterministic without an API key.** Data generation is
   seeded (`SEED` in generate_data.py); tests and the retrieval eval must
   never require network access or a key. CI depends on this.

## Gotchas

- `data/` is **gitignored**. Regenerate it with `python scripts/generate_data.py`
  after cloning; tests generate their own copy in a tmp dir via `conftest.py`.
- Two months (2025-02, 2025-08) exist as BOTH csv and pdf statements. Ingestion
  prefers CSV; the PDF copies exist for the parser cross-validation test
  (`test_pdf_and_csv_parsers_agree`). Keep them in sync if you touch the generator.
- The 50-question count is asserted in `build_questions()`. Adding a question
  means updating that assertion — intentional friction so eval size changes
  are deliberate.
- `test_hybrid_recall_floor` is a CI quality gate (hybrid recall@5 ≥ 0.85 with
  the TF-IDF backend). If a retrieval change trips it, the change regressed
  recall — fix the change, don't lower the floor.
- Default model is `claude-sonnet-4-6`, overridable via env var
  `LEDGER_AUDITOR_MODEL`.
- pdfplumber emits noisy warnings on some platforms; they're harmless.

## Where to add things

- New document type (e.g. utility contract): `docs/EXTENDING.md` §1.
- New statement format/bank parser: `docs/EXTENDING.md` §2.
- New eval questions or planted anomalies: `docs/EXTENDING.md` §3.
- Deeper design rationale: `docs/ARCHITECTURE.md`.
