"""
Run many agent queries in one shot to populate Weave traces / dashboard.

Questions are tuned for this repo's router and tools:
  - Transaction prompts include spend/transactions/last/days/category wording plus a category
    (restaurant, groceries, travel, shopping) and a standalone day number for `_extract_days`.
  - Policy prompts avoid transaction routing tokens (spend, spent, transactions, last, days,
    category) so queries hit `retrieve_docs` + policy chunks in data/docs.txt.

Reproducible spend totals: set TRANSACTION_EVAL_AS_OF (default 2026-05-02).

Usage from repo root:
  python -m evals.run_bulk_weave_traces

Optional:
  WEAVE_PROJECT (default financial_support_agent)
  BULK_WEAVE_SLEEP_SEC — seconds between calls (default 0)
  BULK_WEAVE_LIMIT — max rows to run (for quick tests)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

WEAVE_PROJECT = os.getenv("WEAVE_PROJECT", "financial_support_agent")
DEFAULT_AS_OF = "2026-05-02"
DATA_PATH = Path(__file__).parent / "data" / "bulk_weave_questions.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk Weave traces for financial support agent.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="JSON array of {query, expected_route?}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run at most N questions.")
    args = parser.parse_args()

    if "TRANSACTION_EVAL_AS_OF" not in os.environ:
        os.environ["TRANSACTION_EVAL_AS_OF"] = DEFAULT_AS_OF

    import weave  # noqa: E402

    weave.init(WEAVE_PROJECT)

    from agent import run_agent  # noqa: E402

    rows = json.loads(args.data.read_text(encoding="utf-8"))
    if args.limit is not None:
        rows = rows[: args.limit]

    sleep_sec = float(os.getenv("BULK_WEAVE_SLEEP_SEC", "5"))
    mismatches = 0

    for i, row in enumerate(rows, start=1):
        query = row["query"]
        expected = row.get("expected_route")
        out = run_agent(query)
        route = out["route"]
        if expected is not None and route != expected:
            mismatches += 1
            print(f"[{i}/{len(rows)}] ROUTE MISMATCH expected={expected} got={route}: {query!r}")
        else:
            preview = (out["final_answer"] or "")[:120].replace("\n", " ")
            print(f"[{i}/{len(rows)}] route={route} answer_preview={preview!r}")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    print(f"\nDone. {len(rows)} calls; route mismatches: {mismatches}")


if __name__ == "__main__":
    main()
