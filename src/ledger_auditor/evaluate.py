"""Eval harness.

Two layers, so the project is measurable with or without an API key:

  retrieval eval (no key needed):
      For every question tagged retrieval_eval, did the top-k chunks include
      the section the answer lives in? Reported as recall@k for bm25, dense,
      and hybrid — this is the benchmark table in the README.

  end-to-end eval (needs ANTHROPIC_API_KEY):
      Runs the full agent on all 50 questions and scores:
        - answer correctness: numeric match within tolerance when the question
          has a numeric ground truth, else keyword overlap with expected answer
        - citation accuracy: fraction of answers whose quotes all pass the
          deterministic verbatim check (computed by the verifier)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .retrieval import HybridRetriever


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -------------------------------------------------------------- retrieval eval
def _source_hit(results, expected: dict) -> bool:
    frag = expected["must_contain"].lower()
    return any(r.chunk.doc_id == expected["doc"] and frag in r.chunk.text.lower()
               for r in results)


def retrieval_eval(retriever: HybridRetriever, questions: list[dict],
                   k: int = 5) -> dict:
    modes = ("bm25", "dense", "hybrid")
    eligible = [q for q in questions if q.get("retrieval_eval")]
    out = {"k": k, "n_questions": len(eligible), "recall_at_k": {}, "misses": {}}
    for mode in modes:
        hits, misses = 0, []
        for q in eligible:
            results = retriever.search(q["question"], k=k, mode=mode)
            if all(_source_hit(results, exp) for exp in q["expected_sources"]):
                hits += 1
            else:
                misses.append(q["id"])
        out["recall_at_k"][mode] = round(hits / len(eligible), 3)
        out["misses"][mode] = misses
    return out


# ------------------------------------------------------------- end-to-end eval
_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def score_answer(question: dict, answer_text: str) -> bool:
    """Numeric questions: ground-truth number must appear (within tolerance).
    Text questions: majority of the expected answer's numbers OR keywords
    must appear. The number path exists because keyword overlap alone
    false-negatived a correct answer (Q41) whose wording diverged from the
    expected phrasing while containing the exact discriminating amounts."""
    if question.get("numeric") is not None:
        target, tol = float(question["numeric"]), float(question.get("tolerance", 0.01))
        return any(abs(n - target) <= tol for n in _extract_numbers(answer_text))

    got_nums = _extract_numbers(answer_text)
    exp_nums = _extract_numbers(question["answer"])
    if exp_nums:
        matched = sum(1 for e in exp_nums
                      if any(abs(g - e) <= 0.01 for g in got_nums))
        if matched / len(exp_nums) >= 0.5:
            return True

    expected = question["answer"].lower()
    words = [w for w in re.findall(r"[a-z']{4,}", expected)
             if w not in ("that", "this", "with", "from", "only", "after")]
    if not words:
        return expected in answer_text.lower()
    found = sum(1 for w in words if w in answer_text.lower())
    return found / len(words) >= 0.5


def end_to_end_eval(agent, questions: list[dict], verbose: bool = True) -> dict:
    rows = []
    for q in questions:
        ans = agent.ask(q["question"])
        correct = score_answer(q, ans.answer)
        rows.append({"id": q["id"], "category": q["category"], "correct": correct,
                     "verified": ans.verified, "answer": ans.answer,
                     "notes": ans.verification_notes})
        if verbose:
            mark = "PASS" if correct else "FAIL"
            print(f"[{mark}] {q['id']} ({q['category']}) verified={ans.verified}")

    n = len(rows)
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["correct"])
    return {
        "n": n,
        "answer_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "citation_pass_rate": round(sum(r["verified"] for r in rows) / n, 3),
        "accuracy_by_category": {c: round(sum(v) / len(v), 3) for c, v in by_cat.items()},
        "rows": rows,
    }
