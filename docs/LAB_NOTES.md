# Lab notes — the eval-driven debugging log

A chronological record of every live run, what went wrong, the diagnosis, and
the fix. Companion to the condensed "Failure modes" section in the README.
Commit hashes refer to this repo's history.

---

## Phase 1 — Construction (commit `92a89d7`)

Built offline: data generator, ingestion, retrieval, agent, verifier, eval
harness, 14 tests. Everything deterministic passed before any API call was
ever made. Retrieval benchmark with the TF-IDF fallback backend looked fine
(hybrid recall@5 = 1.0, floor test in CI).

**Lesson pre-loaded into the design:** the two eval layers (retrieval-only,
end-to-end) exist so the system is measurable before spending API money.

---

## Phase 2 — Retrieval benchmark with real embeddings (commit `35c41af`)

First run with `sentence-transformers/all-MiniLM-L6-v2` on a real machine:

| recall@k | BM25 | MiniLM | hybrid |
|----------|------|--------|--------|
| @1 | 0.806 | 0.806 | **0.742** |
| @3 | 1.000 | 0.839 | 0.935 |
| @5 | 1.000 | 0.935 | 1.000 |

**Observation:** dense retrieval *lost* to BM25, and hybrid paid a "fusion
tax" at k=1 (worse than either ranker alone).

**Diagnosis:** the eval questions share vocabulary with the contracts, so
BM25 is strong; MiniLM's misses were rare proper nouns ("Granite Shield",
"Illinois") and dollar-figure clauses. RRF averages rank disagreement rather
than resolving it, so a confident-but-wrong dense ranking pushes the right
chunk out of the top-1.

**Action:** no code change — reported honestly in the README instead. A
benchmark where the fancy method loses and you can explain why beats one
where it conveniently wins.

---

## Phase 3 — First live agent run (`ask`) — verifier crash (commit `b422c1b`)

**Run:** "Did the July 2025 rent increase comply with the lease?"

**Agent behavior:** correct answer, correct math via calculator, three
verbatim citations, self-corrected a wrong `category` filter. Good.

**Failure:** `verified: False (verifier error: Expecting value: line 1
column 1)`. The stage-2 verifier model wrapped its JSON verdict in markdown
code fences; `json.loads` choked on the ```` ```json ```` prefix.

**Fix:** `_extract_json()` — pull the first `{...}` object out of the text.
Regression test added. *(This fix was later superseded entirely — see
Phase 8.)*

---

## Phase 4 — First full audit — truncation crash (commit `1607fea`)

**Run:** `ledger-auditor audit -v`

**Failure:** `KeyError: 'answer'` after ~40 tool calls. The trace showed the
agent repeating identical `calculate` calls near the end.

**Diagnosis:** the full audit's final answer (6 findings, tables) exceeded
the `max_tokens=2000` per-turn budget. The API truncated the response
mid-`submit_answer` tool call, producing a tool input with no `answer`
field; the loop crashed on it. The repeated calculator calls were the model
retrying after each truncated attempt.

**Fix:** budget 2000 → 8000, and a malformed `submit_answer` now returns an
`is_error` tool result ("submit again, more concisely") instead of crashing.

**Note:** the *investigation* in this run was already perfect — all six
anomalies located, all math correct, including the $222 rent overcharge.

---

## Phase 5 — Audit run 2 — two silent quality failures (commit `bb8f7b6`)

**Run:** `ledger-auditor audit -v` (now completes without crashing)

**Failure A — false negative:** "Duplicate charges: none confirmed." The
planted duplicate (CLEANCO, 2× $120.00, 2025-03-14) was missed.

**Failure B — miscount:** STREAMAX price-creep months counted as 6 (actual:
5), producing $33 instead of $30.

**Shared root cause:** `query_transactions` returns at most 40 rows of 306.
The agent was *reasoning over a keyhole view* — it literally could not scan
for duplicates, and counting months from raw rows invited error.

**Fix (per architecture rule #1 — the LLM never does the math):**
- `find_duplicates` tool: exact-duplicate scan over ALL rows in the store
- `group_by` on `query_transactions`: deterministic buckets (count, sum,
  date range) per amount/description/category/month
- system prompt: never count rows manually; always use these
- 2 regression tests pinning both exact failures

---

## Phase 6 — Audit run 3 — semantics and verifier scope (commit `12f2634`)

**Run:** all 6 anomalies found; `find_duplicates` caught CLEANCO instantly;
STREAMAX now 5/3 months → correct $30.

**Failure A — wrong quantity:** Finding 1 reported the *full* rent increase
($92.50 × 6 = $555) as the "overcharge" instead of charged-minus-allowed
($37 × 6 = $222). The calculator can't prevent choosing the wrong expression.

**Failure B — verifier scope:** `verified: False` because the verifier
demanded document quotes for transaction-derived findings (duplicates, price
creep) — which cannot have quotes by design; they come from the store.

**Fixes (both prompts):** system prompt now defines overcharge = charged −
contractual maximum and requires stated totals to equal per-unit × count;
verifier prompt now exempts transaction claims and adds an internal
arithmetic-consistency check.

**Also observed:** verifier correctly flagged an uncited "signed January 12,
2024" claim — the discipline working as intended.

---

## Phase 7 — Audit run 4 + full eval run 1 (commit `b2a528c`)

**Audit:** 6/6 findings, correct $222, `verified: True`. Residual wart:
STREAMAX summary mixed two accounting bases ($26 = neither step-accounting's
$21 nor vs-baseline's $30). Detection perfect; summary arithmetic wobbly.

**Eval:** answer accuracy **0.98** (49/50), citation pass rate **0.96**.
Three imperfections, three distinct root causes:

| Item | Symptom | Root cause |
|------|---------|-----------|
| Q41 | FAIL (answer was actually correct) | scorer false negative: keyword overlap missed divergent wording that contained the exact discriminating amounts |
| Q25 | verified=False | verifier exhausted max_tokens=300 on preamble ("Let me verify...") before emitting JSON |
| Q29 | verified=False | verifier over-strictness: failed deterministic transaction claims as "unconfirmed" |

**Fixes:** scorer also matches on the expected answer's numbers (≥50%);
verifier told to *assume* transaction claims; and — fatefully — an
assistant-prefill `{` to force JSON from the first token.

---

## Phase 8 — Eval run 2 — the prefill disaster (commit `54815ad`)

**Run:** citation pass rate collapsed 0.96 → **0.24**. Accuracy unchanged.

**Pattern in the wreckage:** every question that submits citations failed;
the only `verified=True` rows were transaction-only questions that skip
stage 2. All 38 failures were the same API 400: **the model does not support
assistant prefill.**

**Why tests didn't catch it:** nothing in the unit suite touches the live
API. The prefill change passed all 18 tests and failed on its first real
call.

**Fix — eliminate the failure class, don't patch it:** the verdict now comes
back through a **forced tool call** (`tool_choice`), so its shape is
guaranteed by the API. All free-text JSON parsing deleted, including Phase
3's `_extract_json` and its test — that entire category of bug is now
impossible rather than handled.

**Lesson:** some failure modes only end-to-end evals catch. This is the
single best story in the repo.

---

## Phase 9 — Eval run 3 — stable (commits `fb3fedb`, `f9be1a3`)

| Metric | Run 1 | Run 2 (prefill bug) | Run 3 |
|--------|-------|--------------------|-------|
| Answer accuracy | 0.98 | 0.98 | 0.98 |
| Citation pass rate | 0.96 | 0.24 | 0.96 |

The one FAIL per run is always the same family: Q36 (run 3) / Q45 (run 2) —
the overcharge-semantics slip from Phase 6, reduced by prompting from
consistent to ~1-in-50 but not eliminated. Documented in the README as the
known failure mode rather than hidden.

---

## Recurring themes

1. **Deterministic checks before LLM checks.** The verbatim quote gate and
   the store's aggregation caught or prevented more errors than any prompt.
2. **When the agent fails, look for a keyhole.** Both Phase-5 failures came
   from the agent seeing a truncated view of the data. Give tools that
   compute the answer; don't make the model reconstruct it.
3. **Prompts mitigate; tools eliminate.** The overcharge slip (prompt fix)
   still recurs at ~1-in-50. The JSON-parsing bug (tool-choice fix) can
   never recur. Prefer fixes that make the failure impossible.
4. **Evaluate the evaluator.** One of the 50 "failures" was the scorer's
   fault, and one verifier outage was a token budget. The harness is code
   too, and it has bugs too.
5. **Unit tests prove logic; evals prove the system.** The prefill bug is
   the canonical example: green suite, broken product.
