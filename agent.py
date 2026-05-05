from typing import Any, Literal
import weave
from llm import generate_response
from rag import retrieve_docs
from tools import get_spend


DEFAULT_CATEGORY = "groceries"
DEFAULT_DAYS = 30

TRANSACTION_KEYWORDS = [
    "spend",
    "spent",
    "transactions",
    "last",
    "days",
    "category",
]

POLICY_HINTS = ["overdraft", 
    "interest", 
    "apr",
    "dispute",
]


def _extract_category(query: str) -> str:
    lowered = query.lower()
    categories = ["restaurant", "groceries", "travel", "shopping"]
    for category in categories:
        if category in lowered:
            return category
    return DEFAULT_CATEGORY


def _extract_days(query: str) -> int:
    for token in query.split():
        if token.isdigit():
            return int(token)
    return DEFAULT_DAYS

@weave.op()
def _route(query: str) -> dict[str, Any]:
    """Route + confidence in [0, 1]. Lower when txn + policy hints overlap.

    Returns a dict (not a tuple) so Weave traces show named output fields.
    """
    lowered = query.lower()
    has_txn = any(k in lowered for k in TRANSACTION_KEYWORDS)
    has_policy = any(k in lowered for k in POLICY_HINTS)
    ambiguous = has_txn and has_policy

    if has_txn:
        route: Literal["transactions", "rag"] = "transactions"
        reason = "keyword match on spending/transactions"
        confidence = 0.35 if ambiguous else 0.9
    else:
        route = "rag"
        reason = "policy or explanation question detected"
        confidence = 0.85

    return {"route": route, "reason": reason, "confidence": confidence}


def _context_from_transaction(tool_output: dict) -> str:
    return (
        "Transaction Summary:\n"
        f"- Category: {tool_output['category']}\n"
        f"- Days: {tool_output['days']}\n"
        f"- Total spend: ${tool_output['total_spent']:.2f}"
    )


def _context_from_rag(retrieval: dict) -> str:
    parts = [d["text"] for d in retrieval["docs"]]
    return "Retrieved Policy Context:\n" + "\n\n".join(parts)

@weave.op()
def run_agent(query: str) -> dict[str, Any]:
    """Run the agent; tags Weave traces with `router_confidence` for UI monitors."""
    routed = _route(query)
    route = routed["route"]
    router_reason = routed["reason"]
    router_confidence = routed["confidence"]

    tool_used: str | None = None
    tool_input: dict | None = None
    tool_output: dict | None = None
    retrieved_docs: list | None = None

    if route == "transactions":
        category = _extract_category(query)
        days = _extract_days(query)
        tool_used = "get_spend"
        tool_input = {"category": category, "days": days}
        tool_output = get_spend(category=category, days=days)
        context = _context_from_transaction(tool_output)
    else:
        retrieval = retrieve_docs(query)
        retrieved_docs = retrieval["docs"]
        context = _context_from_rag(retrieval)

    final_answer = generate_response(context=context, query=query)

    return {
        "query": query,
        "route": route,
        "router_reason": router_reason,
        "router_confidence": router_confidence,
        "tool_used": tool_used,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "retrieved_docs": retrieved_docs,
        "final_answer": final_answer,
    }


def answer_query(query: str) -> str:
    """Backward-compatible: final answer text only."""
    return run_agent(query)["final_answer"]
