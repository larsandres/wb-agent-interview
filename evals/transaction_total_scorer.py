"""
Weave Scorer: GPT judge decides if the agent answer is consistent with ground-truth spend facts.
"""

from __future__ import annotations

import json
import os
from typing import Any

import weave
from openai import OpenAI
from pydantic import Field

from weave import Scorer


class TransactionOutputJudgeScorer(Scorer):
    """LLM-as-judge: compares assistant output to dataset ground truth; returns is_true + reasoning."""

    model_id: str = Field(
        default_factory=lambda: os.getenv(
            "TXN_SCORER_MODEL", "gpt-5-nano-2025-08-07"
        )
    )
    system_prompt: str = (
        "You verify whether a financial assistant's answer is correct with respect to the "
        "authoritative ground truth. Focus on whether stated totals, categories, and time windows "
        "match the facts. Respond with JSON only."
    )

    @weave.op()
    def score(
        self,
        *,
        output: str,
        query: str,
        gold_total: float,
        gold_reference: str,
    ) -> dict[str, Any]:
        """Judge whether `output` is true relative to ground truth.

        Args:
            output: Agent final answer.
            query: User question from the dataset.
            gold_total: Canonical total (USD) for the evaluated window/category.
            gold_reference: Canonical natural-language fact string.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key)

        user_msg = (
            f"User question:\n{query}\n\n"
            "Ground truth (authoritative):\n"
            f"- Expected total (USD): {gold_total}\n"
            f"- Reference: {gold_reference}\n\n"
            f"Assistant response:\n{output}\n\n"
            "The response must be a JSON object with the following fields:\n"
            "- is_true: a boolean stating whether the output is true or false based on the "
            "ground truth (totals, category, and time window must align).\n"
            "- reasoning: your reasoning as to why the statement is true or false."
        )

        res = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        raw = res.choices[0].message.content or "{}"
        data = json.loads(raw)
        is_true = bool(data.get("is_true"))
        reasoning = data.get("reasoning")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning) if reasoning is not None else ""
        return {"is_true": is_true, "reasoning": reasoning}
