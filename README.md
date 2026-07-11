# ledger-auditor

**An agentic RAG system that doesn't just answer questions about your documents — it audits them against each other.**

Point it at a folder of personal records (lease, insurance policy, a year of bank statements) and ask:

> *"Did my July rent increase comply with my lease?"*

The agent retrieves the lease's increase-cap clause, pulls the actual rent transactions from your statements, computes the allowed maximum in a deterministic calculator tool, and answers with verbatim citations — which a second verification pass checks against the source documents before anything is returned.

```
$ ledger-auditor ask "Did the July 2025 rent increase comply with the lease?"

No. Rent rose from $1,850.00 to $1,942.50 in July 2025 — a 5.0% increase.
The lease caps increases at 3% ($1,905.50 max), so you are being overcharged
$37.00/month ($222.00 across July–December).

  [lease] "Any increase shall not exceed 3% of the then-current base rent"
verified: True
```

## Why this isn't another chat-with-your-docs project

Tools like PrivateGPT, Khoj, and AnythingLLM do retrieval → answer. This project targets the harder problem: **cross-document reconciliation**, where the answer doesn't exist in any single chunk. It requires multi-step decomposition (which contract clause governs this? which transactions are relevant?), structured queries alongside semantic search, delegated arithmetic, and verification that the final claim is actually entailed by the sources.

## Architecture

```
                       ┌─────────────────────────────────────────┐
 PDFs (lease,          │              AUDIT AGENT                │
 insurance) ──┐        │  (Claude tool-use loop, ≤12 turns)      │
              ▼        │                                         │
   heading-aware       │  search_documents ──► HybridRetriever   │
   chunking            │      (BM25 + dense, RRF fusion)         │
              │        │  query_transactions ► TransactionStore  │
 CSV/PDF      ▼        │      (filter / count / sum)             │
 statements ──► normal-│  calculate ─────────► AST-whitelisted   │
              ization  │      (LLM never does math)              │
                       │  submit_answer ─────► verbatim quotes   │
                       └───────────────┬─────────────────────────┘
                                       ▼
                       ┌─────────────────────────────────────────┐
                       │              VERIFIER                   │
                       │ 1. deterministic: every quote must      │
                       │    literally exist in the cited doc     │
                       │    (catches fabricated citations free)  │
                       │ 2. LLM entailment: do the quotes        │
                       │    support every claim in the answer?   │
                       └─────────────────────────────────────────┘
```

Design decisions worth noting:

- **Heading-aware chunking.** Contracts are chunked one section per chunk, never splitting a clause. Splitting "Landlord may charge a late fee of $50.00" across two chunks is the #1 cause of bad contract RAG.
- **Structured data stays structured.** Transactions are never embedded — 306 bank rows in a vector index is how you get hallucinated sums. They live in a typed store the agent queries with filters; the store returns counts and sums computed in Python.
- **The LLM never does arithmetic.** A `calculate` tool evaluates expressions through an AST whitelist (no `eval`), and the system prompt forbids mental math.
- **Citations are verbatim or rejected.** The verifier's first stage is a zero-cost string check: a quote that doesn't literally appear in the cited document marks the answer unverified before any LLM judging happens.

## Quickstart

```bash
pip install -e ".[dense,dev]"          # or just `pip install -e .` for the CI-light setup
python scripts/generate_data.py        # builds the synthetic demo dataset
ledger-auditor ingest                  # parse everything, print corpus stats

# no API key needed:
ledger-auditor eval --mode retrieval   # retrieval benchmark
pytest                                 # 14 tests incl. retrieval quality gates

# with ANTHROPIC_API_KEY set:
ledger-auditor ask "Was the April late fee charged correctly?"
ledger-auditor audit                   # full sweep: finds all 6 planted anomalies
ledger-auditor eval --mode full        # 50-question end-to-end eval
```

## The dataset

Everything runs on a fully synthetic, deterministic persona (`scripts/generate_data.py`): a lease PDF, an insurance declarations PDF, and 12 months of bank statements (CSV, plus two months rendered as PDF tables to exercise PDF table extraction). Six anomalies are planted with known ground truth:

| # | Anomaly | Money at stake |
|---|---------|----------------|
| F1 | Rent raised 5% — lease caps increases at 3% | $37.00/mo, $222.00 total |
| F2 | $75 late fee charged — lease allows $50 | $25.00 |
| F3 | Pet fee raised to $45 — lease fixes it at $35 | $40.00 |
| F4 | Duplicate $120 charge, same vendor, same day | $120.00 |
| F5 | Insurance autopay rose mid-term — policy allows changes only at renewal | $25.00 |
| F6 | Subscription price creep ($12.99 → $15.99 → $17.99) | $30.00 |

Because the generator also computes ground truth, the **50-question eval set is derived from the same constants** — answers can't drift out of sync with the data.

## Evaluation

Two layers, so the project is measurable with or without an API key.

**Retrieval benchmark** (no key needed; section-level recall on the 31 document-grounded questions). Two dense backends: MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, the real one) and the dependency-light TF-IDF fallback used in CI:

| recall@k | BM25 | MiniLM | hybrid (BM25+MiniLM) | TF-IDF | hybrid (BM25+TF-IDF) |
|----------|------|--------|----------------------|--------|-----------------------|
| @1 | 0.806 | 0.806 | 0.742 | 0.774 | 0.774 |
| @2 | 0.935 | 0.839 | 0.935 | 0.903 | 0.903 |
| @3 | **1.000** | 0.839 | 0.935 | 0.935 | **1.000** |
| @5 | 1.000 | 0.935 | 1.000 | 1.000 | 1.000 |

The honest finding: **on this corpus, BM25 alone wins.** The eval questions share vocabulary with the contract language, and dense retrieval's characteristic failures show up exactly where you'd predict — rare proper nouns ("Granite Shield", "Illinois") that embeddings underweight but BM25 matches exactly, and dollar-figure clauses where semantic similarity is diffuse. Fusion mostly recovers BM25's answers but pays a small tax at low k when a confident-but-wrong dense ranking pushes the right chunk down (hybrid@1: 0.742 < both rankers — RRF averages disagreement instead of resolving it). Hybrid's expected payoff — paraphrased queries lexically distant from the source ("how much did I put down when I moved in?") — is underrepresented in this eval set, which is a measured limitation, not a hidden one. The retrieval eval is wired into CI as a quality gate: `test_hybrid_recall_floor` fails the build if hybrid recall@5 drops below 0.85.

**End-to-end eval** (`eval --mode full`) runs the agent on all 50 questions and reports:

- *answer accuracy* — numeric ground truth within tolerance, keyword match otherwise
- *citation pass rate* — fraction of answers whose every quote survives the verbatim check
- accuracy broken down by category (lookup / aggregate / cross-document audit)

## Failure modes observed (and what was done)

- **Fabricated citations.** Early versions paraphrased quotes. Fixed by making the verbatim check a hard gate and instructing the agent that paraphrased quotes will be rejected.
- **LLM arithmetic drift.** Sums over 40+ transactions were unreliable. Fixed by returning `count`/`sum` from the transaction store itself and banning mental math.
- **Clause splitting.** Fixed-window chunking split the late-fee clause from its dollar amount. Fixed with heading-aware chunking; `test_chunks_are_section_scoped` guards it.
- **PDF table extraction.** `pdfplumber` row order is not guaranteed across engines; the CSV/PDF cross-validation test (`test_pdf_and_csv_parsers_agree`) pins parser agreement on identical months.

## Repo layout

```
scripts/generate_data.py     synthetic dataset + ground truth + 50-question eval set
src/ledger_auditor/
  ingest.py                  PDF/CSV parsing, heading-aware chunking, normalization
  retrieval.py               BM25 + pluggable dense backend + RRF hybrid
  tools.py                   agent tools: search, transaction store, safe calculator
  agent.py                   tool-use loop + two-stage verifier
  evaluate.py                retrieval + end-to-end eval harness
  cli.py                     ingest / ask / audit / eval
tests/                       14 tests incl. retrieval quality gates and parser cross-validation
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data flow, retrieval and fusion rationale, agent loop, verifier design, eval methodology
- [docs/EXTENDING.md](docs/EXTENDING.md) — adding document types, bank parsers, eval questions, dense backends
- [CLAUDE.md](CLAUDE.md) — orientation file for AI coding agents (commands, code map, architectural invariants)

## Limitations

Synthetic data is a controlled environment — real bank PDFs are far messier (multi-line descriptions, OCR noise, balance columns). The ingestion layer is built to be extended per-institution. The verifier checks that quotes exist and support the answer; it does not yet check that the answer is *complete* (recall of findings), which is what `ledger-auditor audit` plus the ground-truth findings file measures manually.

## License

MIT
