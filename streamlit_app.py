import os
import uuid

from dotenv import load_dotenv
import streamlit as st
import weave

load_dotenv()
weave.init(os.getenv("WEAVE_PROJECT", "financial_support_agent"))

from agent import run_agent  # noqa: E402

INITIAL_MESSAGES = [
    {
        "role": "assistant",
        "content": "Hi! Ask me about spending by category or policy questions.",
    }
]


def _ensure_weave_thread_id() -> None:
    if "weave_thread_id" not in st.session_state:
        st.session_state.weave_thread_id = str(uuid.uuid4())


def _reset_conversation() -> None:
    st.session_state.messages = [dict(m) for m in INITIAL_MESSAGES]
    st.session_state.weave_thread_id = str(uuid.uuid4())


def _run_agent_in_thread(query: str) -> dict:
    """Group each turn under the same Weave thread until clear / new session."""
    _ensure_weave_thread_id()
    with weave.thread(st.session_state.weave_thread_id):
        return run_agent(query)


st.set_page_config(page_title="Financial Support Agent", page_icon="💬")
head_l, head_r = st.columns([4, 1])
with head_l:
    st.title("Financial Support Agent")
    st.caption("Ask about spending by category or financial support policies.")
with head_r:
    st.write("")  # align button below title row
    if st.button("Clear conversation", use_container_width=True):
        _reset_conversation()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [dict(m) for m in INITIAL_MESSAGES]
_ensure_weave_thread_id()

with st.sidebar:
    st.markdown("**Weave thread**")
    st.caption(
        "All turns in this chat share this `thread_id` until you clear the conversation."
    )
    st.code(st.session_state.weave_thread_id, language="text")

st.subheader("Suggested Queries")
c1, c2, c3 = st.columns(3)
quick_queries = [
    ("Spend on groceries in the last 60 days?", "What did I spend on groceries in the last 60 days?"),
    ("What is the overdraft policy?", "What is the overdraft policy?"),
    ("How can I dispute travel transactions?", "How can I dispute travel transactions?"),
]
for col, (label, query) in zip((c1, c2, c3), quick_queries):
    with col:
        if st.button(label, use_container_width=True):
            with st.spinner("Getting a response..."):
                result = _run_agent_in_thread(query)
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append(
                {"role": "assistant", "content": result["final_answer"]}
            )
            st.rerun()

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your question...")
if user_input:
    with st.spinner("Getting a response..."):
        result = _run_agent_in_thread(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append(
        {"role": "assistant", "content": result["final_answer"]}
    )
    st.rerun()
