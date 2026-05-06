from pathlib import Path
import weave

DOCS_PATH = Path(__file__).parent / "data" / "docs.txt"
TOP_K = 3


def _load_chunks() -> list[str]:
    text = DOCS_PATH.read_text(encoding="utf-8").strip()
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

@weave.op(kind="TOOL")
def retrieve_docs(query: str) -> dict:
    """
    Return top 2–3 keyword-scored chunks with explicit scores.
    Schema: {"query": str, "docs": [{"text": str, "score": float}, ...]}
    """
    query_terms = set(query.lower().split())
    chunks = _load_chunks()

    scored: list[tuple[float, str]] = []
    for chunk in chunks:
        chunk_terms = set(chunk.lower().split())
        score = float(len(query_terms & chunk_terms))
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    positive = [(s, c) for s, c in scored if s > 0][:TOP_K]

    if not positive:
        # Fallback: best-effort single chunk
        chosen = scored[0][1] if scored else ""
        docs = [{"text": chosen, "score": 0.0}] if chosen else []
    else:
        docs = [{"text": chunk, "score": score} for score, chunk in positive]

    return {"query": query, "docs": docs}
