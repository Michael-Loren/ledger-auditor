# Extending ledger-auditor

Recipes for the three most likely changes. Follow the architectural rules in
CLAUDE.md — especially: structured data stays structured, and the eval set is
generated, never hand-edited.

## 1. Add a new document type (e.g. a utility contract)

1. **Generator**: add a `write_utility_pdf()` in `scripts/generate_data.py`
   modeled on `write_lease_pdf()`. Give sections clear headings.
2. **Chunking**: check the headings match `_HEADING_RE` in
   `src/ledger_auditor/ingest.py`; extend the regex if not. Run
   `ledger-auditor ingest` and confirm the chunk count rose and sections look
   right (`ingest_docs` derives `doc_id` from the filename stem).
3. **Eval**: add lookup questions for the new doc in `build_questions()` with
   `sources=[R("utility", "<distinctive fragment>")]` and
   `retrieval_eval=True`. Update the `assert len(q) == 50`.
4. **Anomaly (optional)**: plant a mismatch between the contract and the
   transactions, add it to `build_ground_truth()`, and add an audit question.
5. Run `pytest` — the recall floor test will tell you if the new sections are
   retrievable.

## 2. Add a real bank-statement parser

Real bank exports vary wildly; add one parser per institution.

1. Write `_parse_<bank>_statement(path) -> list[Transaction]` in `ingest.py`.
   Normalize into the existing `Transaction` fields (ISO dates, negative
   debits). Put institution detection (filename pattern or header sniff) in
   `ingest_statements`.
2. Cross-validate: if you can export the same period in two formats, add a
   test like `test_pdf_and_csv_parsers_agree` asserting identical totals.
3. Never loosen `Transaction` typing to accommodate a messy source — fix it
   in the parser.

## 3. Add eval questions or planted anomalies

- All ground truth lives in `scripts/generate_data.py` constants. To plant an
  anomaly: add the constant, apply it in `build_transactions()` (or the doc
  writers), describe it in `build_ground_truth()`, and add question(s) whose
  answers are **computed from the constant**, not typed as literals.
- Question fields:
  - `numeric` + `tolerance`: enables strict numeric scoring.
  - `expected_sources` + `retrieval_eval: true`: includes the question in the
    retrieval benchmark. `must_contain` should be a fragment that appears
    ONLY in the correct section.
  - `category`: `lookup` | `aggregate` | `audit` (drives the per-category
    accuracy breakdown).
- Update the count assertion in `build_questions()` deliberately.
- Regenerate (`python scripts/generate_data.py`) and run
  `ledger-auditor eval --mode retrieval` to confirm the new questions retrieve.

## 4. Swap or add a dense backend

Implement the two-method protocol in `retrieval.py`:

```python
class MyBackend:
    name = "mybackend"
    def fit(self, texts: list[str]) -> None: ...
    def scores(self, query: str) -> np.ndarray: ...   # one score per chunk
```

Pass it to `HybridRetriever(chunks, dense_backend=MyBackend())`, or wire a
CLI choice into `_build()` in `cli.py`. Keep `TfidfBackend` as the CI default
— CI must never download models.
