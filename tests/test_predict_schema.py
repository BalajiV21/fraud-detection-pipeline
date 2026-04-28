"""Schema test for the serving Transaction model."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving-service"))

from app.main import Transaction  # noqa: E402


def test_transaction_accepts_required_only():
    t = Transaction(TransactionAmt=12.34)
    d = t.model_dump()
    assert d["TransactionAmt"] == 12.34


def test_transaction_accepts_extra_fields():
    t = Transaction(TransactionAmt=12.34, V1=0.1, C1=2, id_30="Windows 10")
    d = t.model_dump()
    assert d["V1"] == 0.1
    assert d["C1"] == 2
    assert d["id_30"] == "Windows 10"
