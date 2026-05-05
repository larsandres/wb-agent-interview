"""
RAG retrieval evaluation using Weave Evaluation + native EmbeddingSimilarityScorer.

Compares concatenated top-k retrieved chunks to a gold policy snippet (embedding cosine).

Requires:
  - **W&B login**: `weave.init` talks to api.wandb.ai — run `wandb login` once, or set `WANDB_API_KEY`
    in `.env` (loaded automatically). A **401 Unauthorized** on graphql almost always means missing/invalid W&B auth.
  - `WEAVE_PROJECT` (optional): defaults to `financial_support_agent`
  - `WEAVE_RAG_EVAL_DATASET` (optional): Weave dataset object name; default `rag_retrieval_eval_golden`
  - `OPENAI_API_KEY` for EmbeddingSimilarityScorer (default embedding model via litellm)

Run from repo root:
  python -m evals.run_rag_eval

Swap in your own scorer later by editing SCORERS below.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo root on sys.path when executed as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import weave  # noqa: E402
from weave import Dataset  # noqa: E402
from weave.evaluation.eval import Evaluation  # noqa: E402
from weave.scorers import EmbeddingSimilarityScorer  # noqa: E402

from rag import retrieve_docs  # noqa: E402

WEAVE_PROJECT = os.getenv("WEAVE_PROJECT", "financial_support_agent")
DATA_PATH = Path(__file__).parent / "data" / "rag_golden.json"
# Stable Weave object name so this eval's dataset is not merged with other unnamed lists.
RAG_EVAL_DATASET_NAME = os.getenv("WEAVE_RAG_EVAL_DATASET", "rag_retrieval_eval_golden")


@weave.op()
def rag_retrieval_model(query: str) -> str:
    """Evaluated operation: keyword RAG output as a single string for scoring."""
    payload = retrieve_docs(query)
    return "\n\n".join(d["text"] for d in payload["docs"])


def load_examples() -> list[dict]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def main() -> None:
    load_dotenv(_ROOT / ".env")
    weave.init(WEAVE_PROJECT)

    dataset = Dataset(name=RAG_EVAL_DATASET_NAME, rows=load_examples())
    # Native Weave scorer: embedding cosine similarity vs gold_reference (mapped to `target`)
    scorers = [
        EmbeddingSimilarityScorer(
            threshold=float(os.getenv("RAG_EVAL_SIM_THRESHOLD", "0.45")),
            column_map={"target": "gold_reference"},
        ),
    ]

    evaluation = Evaluation(
        dataset=dataset,
        scorers=scorers,
        evaluation_name="rag-retrieval-embedding-similarity",
        metadata={"kind": "rag_retrieval", "scorer": "EmbeddingSimilarityScorer"},
    )

    summary = asyncio.run(evaluation.evaluate(rag_retrieval_model))
    print("Evaluation summary:", summary)


if __name__ == "__main__":
    main()
