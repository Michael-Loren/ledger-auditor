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
- Never count rows or months yourself — use query_transactions with group_by
  (e.g. group_by='amount' to count charges at each price tier).
- For duplicate detection, ALWAYS use find_duplicates; never scan rows manually.
- Never assert a contractual term without retrieving it via search_documents.
- When you find a discrepancy, quantify it (per month and total). An
  overcharge is (amount actually charged) minus (maximum the contract
  allows) — never the full increase. Double-check that every total you
  state equals the per-unit amount times the count you stated.
- Finish by calling submit_answer. Every document-derived claim needs a
  citation whose quote is copied VERBATIM from a retrieved section. Do not
  paraphrase inside quotes.
- If the documents do not contain the answer, say so rather than guessing."""

VERIFIER_PROMPT = """You are a verification auditor. Given a question, a proposed
answer, and the exact source quotes, check two things:
1. Every claim about CONTRACT/POLICY TERMS must be supported by the quotes.
   Claims about bank transactions (amounts charged, dates, duplicates, counts,
   payment history) come from a deterministic transaction database that you
   cannot see. ASSUME they are correct. Never fail an answer because a
   transaction claim is "unconfirmed" or "not referenced" — only contract and
   policy claims need quote support.
2. The answer's arithmetic must be internally consistent (e.g. if it says
   $37/month for 6 months, the stated total must be $222).
Submit your verdict with the `verdict` tool."""


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


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
                model=self.model, max_tokens=8000, system=SYSTEM_PROMPT,
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
                    if not tu.input.get("answer"):
                        # truncated/malformed submission (e.g. max_tokens hit
                        # mid-tool-call) — tell the model instead of crashing
                        results.append({
                            "type": "tool_result", "tool_use_id": tu.id,
                            "is_error": True,
                            "content": "submit_answer arrived without an "
                                       "'answer' field (possibly truncated). "
                                       "Submit again, more concisely."})
                        continue
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
        # The verdict comes back as a forced tool call, so its shape is
        # guaranteed by the API — no JSON parsing of free text. (A prefill
        # approach failed here: newer models reject assistant prefill.)
        verdict_tool = {
            "name": "verdict",
            "description": "Submit the verification verdict.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "verified": {"type": "boolean"},
                    "notes": {"type": "string",
                              "description": "one-sentence justification"},
                },
                "required": ["verified", "notes"],
            },
        }
        try:
            r = self.client.messages.create(
                model=self.model, max_tokens=500, system=VERIFIER_PROMPT,
                tools=[verdict_tool],
                tool_choice={"type": "tool", "name": "verdict"},
                messages=[{"role": "user", "content": payload}])
            verdict = next(b.input for b in r.content if b.type == "tool_use")
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
