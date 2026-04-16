import os

import streamlit as st

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from chatbot_backend import build_legal_chain

# Initialize chatbot
st.title("⚖️ NomoSys – AI Legal Chatbot ")

API_BASE_URL = os.getenv("NOMOSYS_API_URL")


@st.cache_resource
def load_chain():
    return build_legal_chain()


def call_backend_api(question: str, history: list[tuple[str, str]]) -> str:
    if httpx is None:
        raise RuntimeError("Missing dependency: httpx")

    if not API_BASE_URL:
        raise RuntimeError("NOMOSYS_API_URL is not set")

    url = f"{API_BASE_URL.rstrip('/')}/chat"
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            url,
            json={"question": question, "history": history, "translate": True},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", "")


if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a legal question:")

if query:
    try:
        if API_BASE_URL:
            answer = call_backend_api(query, st.session_state.history)
        else:
            qa_chain = load_chain()
            result = qa_chain.invoke({"question": query, "chat_history": st.session_state.history})
            answer = result["answer"]

        st.session_state.history.append((query, answer))
        st.write("**NomoSys:**", answer)
    except Exception as e:
        st.error(
            "Backend is unreachable. If you're hosting on Streamlit Cloud, you must run the LLM backend elsewhere "
            "and set the environment variable NOMOSYS_API_URL to your FastAPI server URL.\n\n"
            f"Details: {type(e).__name__}: {e}"
        )

# Display chat history
if st.session_state.history:
    st.markdown("### Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
