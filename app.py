import os
import re

import streamlit as st

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from chatbot_backend import build_legal_chain, translate_answer

# Initialize chatbot
st.title("⚖️ NomoSys – AI Legal Chatbot ")

API_BASE_URL = os.getenv("NOMOSYS_API_URL")

SUPPORTED_REPLY_LANGUAGES = {
    "Auto (match my input)": "auto",
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Urdu": "ur",
    "Arabic": "ar",
    "Punjabi": "pa",
}


def detect_input_language(query: str) -> str:
    explicit_match = re.search(r"\bin\s+([A-Za-z]+)\b", query or "", re.IGNORECASE)
    if explicit_match:
        explicit_lang = explicit_match.group(1).lower()
        for label, code in SUPPORTED_REPLY_LANGUAGES.items():
            if label.lower() == explicit_lang:
                return code

    script_patterns = [
        (r"[\u0900-\u097F]", "hi"),
        (r"[\u0C00-\u0C7F]", "te"),
        (r"[\u0B80-\u0BFF]", "ta"),
        (r"[\u0C80-\u0CFF]", "kn"),
        (r"[\u0D00-\u0D7F]", "ml"),
        (r"[\u0980-\u09FF]", "bn"),
        (r"[\u0A00-\u0A7F]", "pa"),
        (r"[\u0600-\u06FF]", "ur"),
        (r"[\u0750-\u077F]", "ar"),
        (r"[\u0A80-\u0AFF]", "gu"),
    ]
    for pattern, language_code in script_patterns:
        if re.search(pattern, query or ""):
            return language_code

    return "en"


def resolve_reply_language(query: str) -> str:
    selected = st.session_state.get("reply_language", "Auto (match my input)")
    selected_code = SUPPORTED_REPLY_LANGUAGES.get(selected, "auto")
    if selected_code != "auto":
        return selected_code
    return detect_input_language(query)


@st.cache_resource
def load_chain():
    return build_legal_chain()


def call_backend_api(question: str, history: list[tuple[str, str]]) -> str:
    if httpx is None:
        raise RuntimeError("Missing dependency: httpx")

    if not API_BASE_URL:
        raise RuntimeError("NOMOSYS_API_URL is not set")

    url = f"{API_BASE_URL.rstrip('/')}/chat"
    timeout = httpx.Timeout(300.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            url,
            json={"question": question, "history": history, "translate": False},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", "")


if "history" not in st.session_state:
    st.session_state.history = []

if "reply_language" not in st.session_state:
    st.session_state.reply_language = "Auto (match my input)"

with st.sidebar:
    st.subheader("Language")
    st.selectbox(
        "Reply language",
        list(SUPPORTED_REPLY_LANGUAGES.keys()),
        key="reply_language",
        help="Ask in English, Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Urdu, Arabic, or Punjabi.",
    )
    st.caption("Set Auto to reply in the same language you type.")

query = st.text_input("Ask a legal question in any supported language:")

if query:
    try:
        if API_BASE_URL:
            answer = call_backend_api(query, st.session_state.history)
        else:
            qa_chain = load_chain()
            result = qa_chain.invoke({"question": query, "chat_history": st.session_state.history})
            answer = result["answer"]

        target_language = resolve_reply_language(query)
        if target_language != "en":
            answer = translate_answer(answer, target_language)

        st.session_state.history.append((query, answer))
        st.write("**NomoSys:**", answer)
    except Exception as e:
        if API_BASE_URL:
            prefix = (
                f"Backend API is unreachable: {API_BASE_URL}. "
                "If you're using Render free tier, the service may be sleeping; retry after it wakes up.\n\n"
            )
        else:
            prefix = (
                "LLM backend is not configured for Streamlit Cloud. Set NOMOSYS_API_URL to your FastAPI server URL, "
                "or set LLM_PROVIDER=openai and OPENAI_API_KEY to run without a separate backend.\n\n"
            )

        st.error(f"{prefix}Details: {type(e).__name__}: {e}")

# Display chat history
if st.session_state.history:
    st.markdown("### Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
