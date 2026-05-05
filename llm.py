import os
import weave
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@weave.op()
def generate_response(context: str, query: str) -> str:
    """
    Response generation via OpenAI chat completions.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Set it in your environment or .env file."
        )

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a financial support assistant. "
        "Use only the provided context. Be concise and clear."
    )
    user_prompt = f"Context:\n{context}\n\nUser question:\n{query}"

    completion = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    return content or ""
