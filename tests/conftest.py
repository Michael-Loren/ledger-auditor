import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """Generate the synthetic dataset once per test session."""
    out = tmp_path_factory.mktemp("data")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_data.py"), "--out", str(out)],
        check=True, cwd=ROOT)
    return out


@pytest.fixture(scope="session")
def retriever_and_store(data_dir):
    from ledger_auditor.ingest import ingest_docs, ingest_statements
    from ledger_auditor.retrieval import HybridRetriever, TfidfBackend
    from ledger_auditor.tools import TransactionStore
    chunks = ingest_docs(data_dir / "docs")
    txns = ingest_statements(data_dir / "statements")
    return HybridRetriever(chunks, dense_backend=TfidfBackend()), TransactionStore(txns)
