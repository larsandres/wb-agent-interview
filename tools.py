import json
import os
import weave
from datetime import date, timedelta
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "transactions.json"


def _load_transactions() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_of_date() -> date:
    """Use TRANSACTION_EVAL_AS_OF (ISO date) when set for reproducible evals; else real today."""
    raw = os.getenv("TRANSACTION_EVAL_AS_OF")
    if raw:
        return date.fromisoformat(raw)
    return date.today()


@weave.op()
def get_spend(category: str, days: int) -> dict:
    """Return structured spend summary for a category over the last N days."""
    transactions = _load_transactions()
    today = _as_of_date()
    window_start = today - timedelta(days=days)

    total = 0.0
    for tx in transactions:
        tx_date = date.fromisoformat(tx["date"])
        if tx["category"] != category:
            continue
        if tx_date < window_start or tx_date > today:
            continue
        total += float(tx["amount"])

    return {
        "category": category,
        "days": days,
        "total_spent": round(total, 2),
    }
