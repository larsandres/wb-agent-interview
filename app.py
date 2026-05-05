import os

from dotenv import load_dotenv
import weave

load_dotenv()
weave.init(os.getenv("WEAVE_PROJECT", "financial_support_agent"))

from agent import answer_query  # noqa: E402


def main() -> None:
    print("Financial Support Agent CLI")
    print("Type your question, or type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = answer_query(query)
        print(f"\nAgent:\n{response}\n")


if __name__ == "__main__":
    main()
