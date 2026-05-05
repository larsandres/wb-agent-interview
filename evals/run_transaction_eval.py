"""
Transaction agent evaluation using Weave Evaluation + TransactionOutputJudgeScorer (GPT judge).

Reproducibility: set `TRANSACTION_EVAL_AS_OF` (ISO date) so `get_spend` uses a fixed "today".
This eval sets it to 2026-05-02 by default so totals match evals/data/transactions_golden.json.

Requires:
  - W&B / Weave auth (same as run_rag_eval)
  - OPENAI_API_KEY for the agent and the judge

Run from repo root:
  python -m evals.run_transaction_eval

Optional: `TXN_SCORER_MODEL` (default `gpt-5-nano-2025-08-07`) for the judge;
  `WEAVE_TRANSACTION_EVAL_DATASET` to override the Weave dataset object name (default `transactions_eval_golden`).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")
WEAVE_PROJECT = os.getenv("WEAVE_PROJECT", "financial_support_agent")

import weave  # noqa: E402
from weave import Dataset  # noqa: E402
from weave.evaluation.eval import Evaluation  # noqa: E402

weave.init(WEAVE_PROJECT)

from agent import run_agent  # noqa: E402
from evals.transaction_total_scorer import TransactionOutputJudgeScorer  # noqa: E402

DATA_PATH = Path(__file__).parent / "data" / "transactions_golden.json"
DEFAULT_AS_OF = "2026-05-02"
# Stable Weave object name so this eval's dataset is not merged with other unnamed lists.
TRANSACTION_EVAL_DATASET_NAME = os.getenv(
    "WEAVE_TRANSACTION_EVAL_DATASET", "transactions_eval_golden"
)


@weave.op()
def transaction_agent_model(query: str) -> str:
    """Evaluated op: natural-language answer only (what users see; what scorers grade)."""
    return run_agent(query)["final_answer"]


def load_examples() -> list[dict]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def main() -> None:
    if "TRANSACTION_EVAL_AS_OF" not in os.environ:
        os.environ["TRANSACTION_EVAL_AS_OF"] = DEFAULT_AS_OF

    dataset = Dataset(name=TRANSACTION_EVAL_DATASET_NAME, rows=load_examples())
    scorers = [TransactionOutputJudgeScorer()]

    evaluation = Evaluation(
        dataset=dataset,
        scorers=scorers,
        evaluation_name="transactions-agent-judge-eval",
        metadata={
            "kind": "transactions",
            "scorer": "TransactionOutputJudgeScorer",
            "as_of_date": os.environ.get("TRANSACTION_EVAL_AS_OF", DEFAULT_AS_OF),
        },
    )

    summary = asyncio.run(evaluation.evaluate(transaction_agent_model))
    print("Evaluation summary:", summary)


if __name__ == "__main__":
    main()
