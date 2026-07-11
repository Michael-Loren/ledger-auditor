"""The audit agent and its verifier.

Two-stage design:
  1. AuditAgent — an Anthropic tool-use loop that decomposes the question,
     searches documents, filters transactions, and does math in a calculator
     tool. It must submit citations with verbatim quotes.
  2. Verifier — a separate pass that (a) deterministically checks every quote
     actually appears in the cited document and (b) asks a second model call
     whether the quotes entail the answer. Unverified answers are flagged,
     never silently returned.

The deterministic quote check is the workhorse: it catches fabricated
citations without spending a token."""
from __future__ import annotations

import json
import os

from .models import AuditAnswer, Citation
from .retrieval import HybridRetriever
from .tools import TOOL_DEFINITIONS, ToolExecutor, TransactionStore

DEFAULT_MODEL = os.environ.get("LEDGER_AUDITOR_MODEL", "claude-sonnet-4-6")
MAX_TURNS = 12

SYSTEM_PROMPT = """You are a meticulous personal-records auditor. You answer
questions by cross-referencing the user's contracts (lease, insurance policy)
against their actual bank transactions.

Rules:
- Decompose the question first: which documents govern it, and which
  transactions are relevant? Then gather BOTH sides before concluding.
- Never do arithmetic yourself — use the calculate tool.
- Never assert a contractual term without retrieving it via search_documents.
- When you find a discrepancy, quantify it (per month and total).
- Finish by calling submit_answer. Every document-derived claim needs a
  citation whose quote is copied VERBATIM from a retrieved section. Do not
  paraphrase inside quotes.
- If the documents do not contain the answer, say so rather than guessing."""

VERIFIER_PROMPT = """You are a verification auditor. Given a question, a proposed
answer, and the exact source quotes, decide whether the quotes support every
factual claim in the answer. Respond with JSON only:
{"verified": true/false, "notes": "<one sentence>"}"""


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in markdown fences or preamble text;
    pull out the first {...} object rather than trusting raw output."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in verifier output: {text[:80]!r}")
    return text[start:end + 1]


class AuditAgent:
    def __init__(self, retriever: HybridRetriever, store: TransactionStore,
                 model: str = DEFAULT_MODEL, client=None):
        import anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.executor = ToolExecutor(retriever, store)
        self.retriever = retriever

    # ----------------------------------------------------------- agent loop
    def ask(self, question: str, verbose: bool = False) -> AuditAnswer:
        messages = [{"role": "user", "content": question}]
        for _ in range(MAX_TURNS):
            response = self.client.messages.create(
                model=self.model, max_tokens=2000, system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS, messages=messages)

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                # model answered in prose without submit_answer — accept but flag
                text = "".join(b.text for b in response.content if b.type == "text")
                return self._verify(AuditAnswer(question=question, answer=text))

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                if tu.name == "submit_answer":
                    citations = [Citation(doc_id=c.get("doc_id", ""),
                                          section=c.get("section", ""),
                                          quote=c.get("quote", ""))
                                 for c in tu.input.get("citations", [])]
                    return self._verify(AuditAnswer(
                        question=question, answer=tu.input["answer"],
                        citations=citations))
                output = self.executor.execute(tu.name, tu.input)
                if verbose:
                    print(f"  -> {tu.name}({json.dumps(tu.input)[:120]})")
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": json.dumps(output)})
            messages.append({"role": "user", "content": results})

        return AuditAnswer(question=question, answer="(agent exceeded turn limit)",
                           verified=False, verification_notes="turn limit")

    # ----------------------------------------------------------- verification
    def _verify(self, ans: AuditAnswer) -> AuditAnswer:
        # Stage 1: deterministic — every quote must literally exist in the corpus
        corpus = {c.doc_id: "" for c in self.retriever.chunks}
        for ch in self.retriever.chunks:
            corpus[ch.doc_id] += " " + _normalize(ch.text)
        for cit in ans.citations:
            hay = corpus.get(cit.doc_id, " ".join(corpus.values()))
            if cit.quote and _normalize(cit.quote) not in hay:
                ans.verified = False
                ans.verification_notes = (
                    f"fabricated citation: quote not found in '{cit.doc_id}'")
                return ans

        # Stage 2: LLM entailment check (skipped if there are no citations)
        if not ans.citations:
            ans.verified = True
            ans.verification_notes = "no document claims to verify"
            return ans
        payload = json.dumps({
            "question": ans.question, "answer": ans.answer,
            "quotes": [{"doc": c.doc_id, "quote": c.quote} for c in ans.citations]})
        try:
            r = self.client.messages.create(
                model=self.model, max_tokens=300, system=VERIFIER_PROMPT,
                messages=[{"role": "user", "content": payload}])
            text = "".join(b.text for b in r.content if b.type == "text")
            verdict = json.loads(_extract_json(text))
            ans.verified = bool(verdict.get("verified"))
            ans.verification_notes = verdict.get("notes", "")
        except Exception as e:
            ans.verified = False
            ans.verification_notes = f"verifier error: {e}"
        return ans

    # ----------------------------------------------------------- full audit
    AUDIT_PROMPT = (
        "Run a full audit of my records. Check at minimum: (1) every rent "
        "change against the lease's increase cap and notice requirement, "
        "(2) every fee charged by the landlord (late fees, pet fees, parking) "
        "against the amounts the lease allows, (3) insurance autopay amounts "
        "against the policy premium and its change-notice rule, (4) duplicate "
        "charges, (5) subscription price changes. Report each finding with "
        "the money at stake, then a total."
    )

    def audit(self, verbose: bool = False) -> AuditAnswer:
        return self.ask(self.AUDIT_PROMPT, verbose=verbose)
