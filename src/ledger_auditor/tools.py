"""Tools exposed to the audit agent.

Design rule: the LLM never does arithmetic and never sees the whole dataset.
It searches documents, filters transactions, and delegates math to a
deterministic calculator — then must cite what it used."""
from __future__ import annotations

import ast
import operator as op

from .models import Transaction
from .retrieval import HybridRetriever

# ----------------------------------------------------------------- calculator
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos, ast.Mod: op.mod}


def safe_calculate(expression: str) -> float:
    """Evaluate an arithmetic expression with an AST whitelist (no eval())."""
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("round", "abs", "min", "max", "sum"):
            fn = {"round": round, "abs": abs, "min": min, "max": max, "sum": sum}[node.func.id]
            return fn(*[_eval(a) for a in node.args])
        if isinstance(node, (ast.List, ast.Tuple)):
            return [_eval(e) for e in node.elts]
        raise ValueError(f"disallowed expression element: {ast.dump(node)}")
    return _eval(ast.parse(expression, mode="eval"))


# ----------------------------------------------------------------- txn engine
class TransactionStore:
    def __init__(self, txns: list[Transaction]):
        self.txns = txns

    def _filter(self, description_contains: str = "", category: str = "",
                date_from: str = "", date_to: str = "",
                min_amount: float | None = None,
                max_amount: float | None = None) -> list[Transaction]:
        rows = []
        for t in self.txns:
            if description_contains and description_contains.lower() not in t.description.lower():
                continue
            if category and t.category != category:
                continue
            if date_from and t.date < date_from:
                continue
            if date_to and t.date > date_to:
                continue
            if min_amount is not None and t.amount < min_amount:
                continue
            if max_amount is not None and t.amount > max_amount:
                continue
            rows.append(t)
        return rows

    def query(self, description_contains: str = "", category: str = "",
              date_from: str = "", date_to: str = "",
              min_amount: float | None = None, max_amount: float | None = None,
              group_by: str = "", limit: int = 40) -> dict:
        rows = self._filter(description_contains, category, date_from, date_to,
                            min_amount, max_amount)
        total = round(sum(t.amount for t in rows), 2)
        out = {
            "count": len(rows),
            "sum_amount": total,
            "rows": [t.to_dict() for t in rows[:limit]],
            "truncated": len(rows) > limit,
        }
        # Aggregation is computed HERE, deterministically — never by the LLM.
        # group_by="amount" answers "how many months at each price tier";
        # group_by="month" answers "what did this cost per month"; etc.
        if group_by in ("amount", "description", "category", "month"):
            buckets: dict = {}
            for t in rows:
                key = t.date[:7] if group_by == "month" else getattr(t, group_by)
                b = buckets.setdefault(key, {"count": 0, "sum_amount": 0.0,
                                             "first_date": t.date, "last_date": t.date})
                b["count"] += 1
                b["sum_amount"] = round(b["sum_amount"] + t.amount, 2)
                b["first_date"] = min(b["first_date"], t.date)
                b["last_date"] = max(b["last_date"], t.date)
            out["groups"] = {str(k): v for k, v in sorted(buckets.items())}
            del out["rows"], out["truncated"]  # groups replace raw rows
        return out

    def find_duplicates(self, date_from: str = "", date_to: str = "") -> dict:
        """Exact-duplicate scan over ALL matching transactions (same date,
        description, and amount). Exists because the agent's row-limited view
        cannot reliably do this itself."""
        seen: dict[tuple, list[Transaction]] = {}
        for t in self._filter(date_from=date_from, date_to=date_to):
            if t.amount < 0:  # only debits are interesting
                seen.setdefault((t.date, t.description, t.amount), []).append(t)
        dups = [{"date": k[0], "description": k[1], "amount": k[2],
                 "occurrences": len(v)}
                for k, v in seen.items() if len(v) > 1]
        return {"duplicate_groups": dups, "n_transactions_scanned":
                sum(len(v) for v in seen.values())}


# --------------------------------------------------------- Anthropic tool defs
TOOL_DEFINITIONS = [
    {
        "name": "search_documents",
        "description": "Semantic + keyword search over the user's documents "
                       "(lease, insurance policy). Returns the most relevant "
                       "sections with doc_id and section heading. Use this for "
                       "anything about contractual terms, limits, or obligations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "doc_filter": {"type": "string",
                               "description": "Optional: restrict to one doc_id, e.g. 'lease'."},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_transactions",
        "description": "Filter the normalized bank-transaction table. Amounts are "
                       "negative for debits. Returns matching rows plus count and "
                       "sum. Use this for anything about actual money movement.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description_contains": {"type": "string"},
                "category": {"type": "string"},
                "date_from": {"type": "string", "description": "ISO date inclusive"},
                "date_to": {"type": "string", "description": "ISO date inclusive"},
                "min_amount": {"type": "number"},
                "max_amount": {"type": "number"},
                "group_by": {
                    "type": "string",
                    "enum": ["amount", "description", "category", "month"],
                    "description": "Aggregate matches into buckets with count/"
                                   "sum/date range, computed deterministically. "
                                   "Use group_by='amount' to count how many "
                                   "charges occurred at each price tier, "
                                   "group_by='month' for per-month totals."},
            },
        },
    },
    {
        "name": "find_duplicates",
        "description": "Scan ALL transactions (not row-limited) for exact "
                       "duplicates: same date, description, and amount. Always "
                       "use this for duplicate detection — the row limit on "
                       "query_transactions makes manual scanning unreliable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "ISO date inclusive"},
                "date_to": {"type": "string", "description": "ISO date inclusive"},
            },
        },
    },
    {
        "name": "calculate",
        "description": "Deterministic arithmetic. ALWAYS use this instead of doing "
                       "math yourself. Supports + - * / ** % round abs min max sum.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer. Every factual claim that comes from "
                       "a document MUST have a citation whose `quote` is copied "
                       "verbatim from a retrieved section.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string"},
                            "section": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["doc_id", "quote"],
                    },
                },
            },
            "required": ["answer"],
        },
    },
]


class ToolExecutor:
    def __init__(self, retriever: HybridRetriever, store: TransactionStore):
        self.retriever = retriever
        self.store = store

    def execute(self, name: str, args: dict):
        if name == "search_documents":
            results = self.retriever.search(
                args["query"], k=args.get("k", 5),
                doc_filter=args.get("doc_filter") or None)
            return [{"doc_id": r.chunk.doc_id, "section": r.chunk.section,
                     "text": r.chunk.text, "rank": r.rank} for r in results]
        if name == "query_transactions":
            return self.store.query(**{k: v for k, v in args.items() if v not in ("", None)})
        if name == "find_duplicates":
            return self.store.find_duplicates(
                date_from=args.get("date_from", ""), date_to=args.get("date_to", ""))
        if name == "calculate":
            try:
                return {"result": safe_calculate(args["expression"])}
            except Exception as e:  # let the model see and correct its mistake
                return {"error": str(e)}
        raise ValueError(f"unknown tool {name}")
