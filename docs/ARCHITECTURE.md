# Architecture

Deeper walkthrough of how ledger-auditor works and why it's built this way.
For a quick orientation see the README; for editing rules see CLAUDE.md.

## 1. Data flow

```
generate_data.py ──► data/docs/*.pdf          (lease, insurance declarations)
                 ──► data/statements/*.csv|pdf (12 months of transactions)
                 ──► data/ground_truth.json    (6 planted anomalies)
                 ──► data/questions.jsonl      (50 eval questions, answers
                                                computed from the same constants)

ingest.py ── extract_pdf_text + chunk_document ──► list[Chunk]
         ── _parse_csv_statement / _parse_pdf_statement ──► list[Transaction]

retrieval.py ── HybridRetriever(chunks) : BM25 index + dense index
tools.py     ── TransactionStore(txns)  : typed filter/aggregate engine

agent.py ── AuditAgent(retriever, store).ask(question) ──► AuditAnswer
                                       .audit()        ──► full sweep
```

## 2. Ingestion

**Documents.** `pdfplumber` extracts text; `chunk_document` splits on
headings matched by `_HEADING_RE` ("3. Rent", "Coverage A — ...", etc.).
One section = one chunk, prefixed with `[section]` so the section name itself
is searchable. Rationale: fixed-window chunking splits clauses — "late fee of"
in one chunk and "$50.00" in the next — which is the single most common
failure in contract RAG.

**Statements.** CSVs parse directly; PDF statements go through
`pdfplumber.extract_tables()` with row-shape validation (date regex on col 0).
When a month exists in both formats, CSV wins and the PDF is used by the test
suite to cross-validate the two parsers (identical row counts and totals).
Transactions become typed `Transaction` records — they are **never** embedded.

## 3. Retrieval

Two rankers over the same chunks:

- **BM25** (`rank_bm25`) on a lowercase token stream that keeps `$`, `%`, `.`
  — dollar amounts and percentages are discriminative tokens in this domain.
- **Dense**, behind a two-method protocol (`fit(texts)`, `scores(query)`):
  - `SentenceTransformerBackend` (all-MiniLM-L6-v2) for real use;
  - `TfidfBackend` (word 1-2gram + char 3-5gram TF-IDF, cosine) for CI, so
    tests never download a model or touch the network.

**Fusion** is Reciprocal Rank Fusion: `score(chunk) = Σ 1/(60 + rank_i)`.
RRF was chosen over weighted score sums because BM25 and cosine scores live
on incomparable scales; rank-based fusion needs no calibration. `search()`
exposes `mode=bm25|dense|hybrid` — not for production use, but so the eval
can benchmark each ranker in isolation (the README table).

## 4. The agent

`AuditAgent.ask()` is a plain Anthropic tool-use loop (max 12 turns, no
framework). The system prompt enforces a decomposition discipline: identify
the governing document AND the relevant transactions before concluding.

Tools (see `tools.py`):

| Tool | Backs onto | Why it exists |
|------|-----------|---------------|
| `search_documents` | HybridRetriever | contractual terms |
| `query_transactions` | TransactionStore | actual money movement; returns count + sum so the model never adds rows itself |
| `calculate` | AST-whitelisted evaluator | deterministic arithmetic; `eval()` is never called |
| `submit_answer` | — | forces structured output: answer + citations with verbatim quotes |

The loop ends when the model calls `submit_answer` (or answers in prose,
which is accepted but flagged). Tool errors (e.g. a malformed calculator
expression) are returned to the model as tool results so it can self-correct.

## 5. Verification

`_verify()` runs two stages, cheapest first:

1. **Deterministic quote check.** Every citation's quote, whitespace- and
   case-normalized, must be a substring of the cited document's chunk text.
   A miss marks the answer `verified: False` with reason
   `fabricated citation` — no tokens spent. This catches the most common
   agent failure (paraphrased or invented quotes) for free.
2. **LLM entailment check.** A second model call receives only the question,
   answer, and quotes (not the full corpus) and returns
   `{"verified": bool, "notes": str}` — do the quotes actually support the
   claims? This catches real quotes used to justify wrong conclusions.

Unverified answers are still returned but flagged; the caller decides what to
do. The CLI prints the flag; the end-to-end eval reports the pass rate.

## 6. Evaluation

Design goal: **measurable without an API key**, fully measurable with one.

- **Retrieval eval** (`retrieval_eval`): for the 31 questions tagged
  `retrieval_eval`, checks whether the top-k chunks include the section the
  answer lives in (doc id + required substring). Run per ranker mode →
  the recall@k table. Wired into CI as a floor assertion.
- **End-to-end eval** (`end_to_end_eval`): runs the agent on all 50 questions.
  Scoring: numeric ground truth within tolerance when present (numbers are
  extracted from the answer text), else ≥50% keyword overlap with the expected
  answer. Reports answer accuracy, citation pass rate, and per-category
  breakdown (lookup / aggregate / audit).

The eval set is generated by the same script and constants that generate the
data, so answers cannot drift out of sync with the corpus. This is why
`data/questions.jsonl` must never be hand-edited.

## 7. Known limitations

- Synthetic statements are cleaner than real bank exports (no multi-line
  descriptions, no OCR noise, no balance column drift).
- The verifier checks groundedness, not completeness: an answer can be
  verified yet miss a finding. Finding-recall is measured manually by
  comparing `audit` output against `data/ground_truth.json`.
- On this small, lexically-aligned corpus BM25 alone saturates recall at k≥3;
  hybrid's value shows on paraphrased queries and larger corpora.
- The two PDF statement months assume the generator's table layout; real
  bank PDFs need per-institution parsers (see docs/EXTENDING.md §2).
