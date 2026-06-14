import os
import re
import logging

import streamlit as st

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from chatbot_backend import build_legal_chain, translate_answer, analyze_case_document, build_combined_chain

logger = logging.getLogger(__name__)

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


# ─── Session State Initialization ─────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "reply_language" not in st.session_state:
    st.session_state.reply_language = "Auto (match my input)"

if "case_summary" not in st.session_state:
    st.session_state["case_summary"] = None

if "case_db" not in st.session_state:
    st.session_state["case_db"] = None

if "case_file_name" not in st.session_state:
    st.session_state["case_file_name"] = None

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Language")
    st.selectbox(
        "Reply language",
        list(SUPPORTED_REPLY_LANGUAGES.keys()),
        key="reply_language",
        help="Ask in English, Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Urdu, Arabic, or Punjabi.",
    )
    st.caption("Set Auto to reply in the same language you type.")

    st.markdown("---")
    st.subheader("📄 Document")
    if st.session_state.get("case_file_name"):
        st.success(f"Loaded: {st.session_state['case_file_name']}")
        if st.button("🗑️ Clear Document"):
            st.session_state["case_db"] = None
            st.session_state["case_summary"] = None
            st.session_state["case_file_name"] = None
            st.rerun()
    else:
        st.caption("No document uploaded.")

# ─── Document Upload Section ──────────────────────────────────────────
with st.expander("📤 Upload a Legal Document for Analysis", expanded=not bool(st.session_state.get("case_summary"))):
    uploaded_file = st.file_uploader(
        "Upload a legal document (FIR, judgment, contract, etc.):",
        type=["pdf", "txt"],
        key="doc_uploader",
        help="Supported formats: PDF, TXT. Max size determined by Streamlit config.",
    )
    if uploaded_file:
        if st.button("📋 Analyze Document", type="primary"):
            with st.spinner("Analyzing document... This may take a minute."):
                try:
                    case_db, summary = analyze_case_document(uploaded_file)
                    st.session_state["case_db"] = case_db
                    st.session_state["case_summary"] = summary
                    st.session_state["case_file_name"] = uploaded_file.name
                    st.success("✅ Document analyzed successfully!")
                    st.rerun()
                except Exception as e:
                    logger.exception(
                        "Failed to analyze uploaded document in Streamlit UI: file_name=%s",
                        getattr(uploaded_file, "name", None),
                    )
                    st.error(f"Failed to analyze document: {type(e).__name__}: {e}")

# ─── Case Summary Display ────────────────────────────────────────────
if st.session_state.get("case_summary"):
    with st.expander("📄 Case Summary", expanded=True):
        st.markdown(st.session_state["case_summary"])

# ─── Query Input ─────────────────────────────────────────────────────
query = st.text_input("Ask a legal question in any supported language:")

if query:
    try:
        if st.session_state.get("case_db"):
            # Dual-RAG: use both case document and legal knowledge base
            combined_chain = build_combined_chain(st.session_state["case_db"])
            result = combined_chain.invoke(
                {"question": query, "chat_history": st.session_state.history}
            )
            answer = result["answer"]
        elif API_BASE_URL:
            answer = call_backend_api(query, st.session_state.history)
        else:
            qa_chain = load_chain()
            result = qa_chain.invoke(
                {"question": query, "chat_history": st.session_state.history}
            )
            answer = result["answer"]

        target_language = resolve_reply_language(query)
        if target_language != "en":
            answer = translate_answer(answer, target_language)

        st.session_state.history.append((query, answer))
        st.write("**NomoSys:**", answer)
    except Exception as e:
        if API_BASE_URL and not st.session_state.get("case_db"):
            prefix = (
                f"Backend API is unreachable: {API_BASE_URL}. "
                "If you're using Render free tier, the service may be sleeping; retry after it wakes up.\n\n"
            )
        else:
            prefix = (
                "An error occurred while processing your question. "
                "Make sure Ollama is running and the model is available.\n\n"
            )
        st.error(f"{prefix}Details: {type(e).__name__}: {e}")

# Display chat history
if st.session_state.history:
    st.markdown("### Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** NomoSys provides legal *information* only, not legal advice. "
    "Always verify important legal references independently and consult a qualified legal professional."
)
