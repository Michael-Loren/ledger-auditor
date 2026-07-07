"""CLI: ledger-auditor {ingest|ask|audit|eval}"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .evaluate import end_to_end_eval, load_questions, retrieval_eval
from .ingest import ingest_docs, ingest_statements
from .retrieval import HybridRetriever, TfidfBackend
from .tools import TransactionStore


def _build(data_dir: Path, backend: str = "auto"):
    chunks = ingest_docs(data_dir / "docs")
    txns = ingest_statements(data_dir / "statements")
    dense = TfidfBackend() if backend == "tfidf" else None
    retriever = HybridRetriever(chunks, dense_backend=dense)
    return retriever, TransactionStore(txns)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ledger-auditor")
    p.add_argument("--data", default="data", help="data directory")
    p.add_argument("--backend", choices=["auto", "tfidf"], default="auto",
                   help="dense backend (tfidf = no model download, used in CI)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="parse everything and print corpus stats")

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("-v", "--verbose", action="store_true")

    audit = sub.add_parser("audit", help="run the full audit")
    audit.add_argument("-v", "--verbose", action="store_true")

    ev = sub.add_parser("eval", help="run the eval harness")
    ev.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ev.add_argument("--k", type=int, default=5)
    ev.add_argument("--out", default=None, help="write JSON results here")

    args = p.parse_args(argv)
    data_dir = Path(args.data)
    retriever, store = _build(data_dir, args.backend)

    if args.cmd == "ingest":
        docs = sorted({c.doc_id for c in retriever.chunks})
        print(f"chunks: {len(retriever.chunks)} from docs {docs}")
        print(f"transactions: {len(store.txns)} "
              f"({store.txns[0].date} .. {store.txns[-1].date})")
        print(f"dense backend: {retriever.dense.name}")
        return

    if args.cmd == "eval" and args.mode == "retrieval":
        results = retrieval_eval(retriever, load_questions(data_dir / "questions.jsonl"),
                                 k=args.k)
        print(json.dumps({k: v for k, v in results.items() if k != "misses"}, indent=2))
        for mode, missed in results["misses"].items():
            if missed:
                print(f"  {mode} missed: {', '.join(missed)}")
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2))
        return

    # everything below needs an API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is required for this command "
                 "(retrieval-only eval works without it: eval --mode retrieval)")

    from .agent import AuditAgent
    agent = AuditAgent(retriever, store)

    if args.cmd == "ask":
        ans = agent.ask(args.question, verbose=args.verbose)
        print(f"\n{ans.answer}\n")
        for c in ans.citations:
            print(f"  [{c.doc_id}] \"{c.quote}\"")
        print(f"\nverified: {ans.verified}  ({ans.verification_notes})")
    elif args.cmd == "audit":
        ans = agent.audit(verbose=args.verbose)
        print(f"\n{ans.answer}\n\nverified: {ans.verified}")
    elif args.cmd == "eval":
        results = end_to_end_eval(agent, load_questions(data_dir / "questions.jsonl"))
        summary = {k: v for k, v in results.items() if k != "rows"}
        print(json.dumps(summary, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
