"""NomoSys Legal Chatbot Backend — Optimized for speed.

Key optimizations vs. original:
- Singleton embeddings + LLM (never re-instantiate)
- Single Gemini call for ALL pages (not per-page)
- 150 DPI instead of 300 DPI for pdf2image
- Gemini output IS the summary (Ollama summary step deleted)
- build_combined_chain() accepts pre-loaded law_db to avoid reload
- Legal-text-aware chunking separators
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None


from langchain_community.document_loaders import TextLoader, PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document

try:
    from langchain_classic.chains import ConversationalRetrievalChain
except Exception:  # pragma: no cover
    from langchain.chains import ConversationalRetrievalChain

logger = logging.getLogger(__name__)

# ─── Singletons ───────────────────────────────────────────────────────────
# Never re-create these. Module-level cache. Thread-safe for reads.

_cached_embeddings = None
_cached_llm = None
_cached_gemini_model = None


def _get_embeddings():
    """Singleton embedding model. Loaded once, reused forever."""
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings
    _cached_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    logger.info("Embedding model loaded (singleton)")
    return _cached_embeddings


def _get_llm():
    """Singleton LLM. Loaded once, reused forever."""
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm
    if ChatOllama is None:
        raise RuntimeError(
            "langchain_ollama is not installed. Run: pip install langchain-ollama"
        )
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    ollama_base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
    kwargs = {}
    if ollama_base_url:
        kwargs["base_url"] = ollama_base_url
    _cached_llm = ChatOllama(model=model, num_ctx=num_ctx, **kwargs)
    logger.info("LLM loaded (singleton): model=%s", model)
    return _cached_llm


def _get_gemini_model():
    """Singleton Gemini model object."""
    global _cached_gemini_model
    if _cached_gemini_model is not None:
        return _cached_gemini_model
    _cached_gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    logger.info("Gemini model loaded (singleton)")
    return _cached_gemini_model


# ─── Prompt Templates ─────────────────────────────────────────────────────

COMBINED_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an advanced legal assistant specializing exclusively in Indian law. "
        "You have access to two sources of information:\n"
        "1. **Case Document Context**: Excerpts from a user-uploaded legal document (FIR, judgment, contract, etc.)\n"
        "2. **Legal Knowledge Context**: Relevant provisions from Indian statutes (Constitution, BNS, CrPC, etc.)\n\n"
        "Both sources are combined in the context below.\n\n"
        "Instructions:\n"
        "Use ONLY Indian law and the provided context. Do NOT invent or hallucinate any law, "
        "section number, article, case fact, document, party, date, or procedural step not supported by the context.\n"
        "If the answer cannot be determined from the provided context, explicitly state that the provided context "
        "does not contain sufficient information to answer confidently.\n"
        "When context contains case-specific facts, reference them directly. When context contains legal provisions, "
        "cite the specific Act, Article, or Section by name. Prefer Bharatiya Nyaya Sanhita, 2023 (BNS) references "
        "over IPC where applicable, but do not guess IPC-to-BNS mappings.\n"
        "Every answer must follow this exact structure:\n\n"
        "⚖️ Legal Analysis\n"
        "Short legal explanation.\n\n"
        "📋 Action Plan\n\n"
        "1. Practical next step\n"
        "2. Practical next step\n"
        "3. Practical next step\n\n"
        "📑 Documents Required\n"
        "• Relevant documents/evidence\n"
        "• Relevant documents/evidence\n\n"
        "📚 Relevant Laws\n"
        "• Relevant Act / Article / Section\n"
        "• Relevant Act / Article / Section\n\n"
        "The Action Plan must be practical and user-focused. Documents Required should list useful evidence/documents "
        "for the issue. If documents are unknown, write: \"Documents depend on the specific facts available.\" "
        "If laws are unavailable, write: \"No specific law identified from available context.\"\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Provide a detailed, well-reasoned answer grounded strictly in the context above."
    )
)

STRICT_LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an advanced constitutional law assistant and legal expert specializing exclusively "
        "in the Constitution of India and Indian laws.\n\n"
        "Instructions:\n"
        "Use ONLY Indian law. Use the given context primarily to answer factually and truthfully. "
        "Do NOT invent or hallucinate any law, section number, article, case fact, document, party, date, or procedural step.\n"
        "If the available context does not contain sufficient information, explicitly say so. If the question is indirectly "
        "related to Indian law, provide a careful Indian-law answer only when you can do so without guessing.\n"
        "Prefer Bharatiya Nyaya Sanhita, 2023 (BNS) references over IPC where applicable, but do not guess IPC-to-BNS mappings. "
        "Never discuss foreign law unless necessary to clarify Indian legal context.\n"
        "Every answer must follow this exact structure:\n\n"
        "⚖️ Legal Analysis\n"
        "Short legal explanation.\n\n"
        "📋 Action Plan\n\n"
        "1. Practical next step\n"
        "2. Practical next step\n"
        "3. Practical next step\n\n"
        "📑 Documents Required\n"
        "• Relevant documents/evidence\n"
        "• Relevant documents/evidence\n\n"
        "📚 Relevant Laws\n"
        "• Relevant Act / Article / Section\n"
        "• Relevant Act / Article / Section\n\n"
        "The Action Plan must be practical and user-focused. Documents Required should list useful evidence/documents "
        "for the issue. If documents are unknown, write: \"Documents depend on the specific facts available.\" "
        "If laws are unavailable, write: \"No specific law identified from available context.\"\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Now provide a detailed and well-reasoned answer relevant ONLY to Indian law and the Constitution of India."
    )
)

# Gemini prompt — sent ONCE for all pages
GEMINI_DOCUMENT_PROMPT = """Analyze this legal document page by page and generate a single consolidated report:

# 📄 Document Overview
- Document Type
- FIR / Case Number
- Police Station / Court
- Date of Filing

# 👥 Parties Involved
- Complainant
- Victim
- Accused

# ⚖ Applicable Laws
List all laws, sections, and acts mentioned.

# 📌 Key Facts
Summarize the incident in bullet points.

# 📅 Important Dates

# 📎 Evidence

# 🔍 Case Strength Assessment
Strong / Moderate / Weak — Explain why.

# 📋 Recommended Immediate Actions

# 📁 Required Supporting Documents

# ⚠ Risk Assessment
- Bail Risk: Not mentioned in the document.
- Evidence Contamination/Loss: Not mentioned in the document.
- Legal Challenges: Not mentioned in the document.
- Organized Crime Link: Not mentioned in the document.
- Dispute over Forest Area: Not mentioned in the document.

# 📝 Executive Summary

# ❓ Suggested Questions
Generate 10 useful questions a user can ask about this case."""


# ─── Language Detection & Translation ─────────────────────────────────────

def detect_output_language(query):
    match = re.search(r"in (\w+)", query, re.IGNORECASE)
    if match:
        lang_word = match.group(1).lower()
        lang_map = {
            "english": "en",
            "hindi": "hi",
            "kannada": "kn",
            "tamil": "ta",
            "telugu": "te",
            "malayalam": "ml",
            "marathi": "mr",
            "bengali": "bn",
            "gujarati": "gu",
            "urdu": "ur",
        }
        return lang_map.get(lang_word, "en")
    return "en"


def translate_answer(answer, target_lang="en"):
    if target_lang == "en":
        return answer
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        translated = GoogleTranslator(source="en", target=target_lang).translate(answer)
        return translated
    except Exception as e:
        logger.warning("Translation error: %s", e)
        return answer


# ─── Legal Knowledge Base ─────────────────────────────────────────────────

def load_legal_docs(folder_path: str | os.PathLike[str] = "data"):
    """Load all .txt and .pdf files from the given folder."""
    folder = Path(folder_path)
    if not folder.is_absolute():
        folder = (Path(__file__).resolve().parent / folder).resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    documents = []
    for file_path in sorted(folder.glob("*")):
        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
    return documents


def _get_legal_splitter():
    """Splitter tuned for legal text: respects Article/Section boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "Article ", "Section ", ". ", " "],
    )


def build_legal_chain():
    """Build the RAG chain for the legal knowledge base. Uses singletons."""
    print("🔍 Loading and preparing legal documents...")
    documents = load_legal_docs("data")
    if not documents:
        raise ValueError("No legal documents found in the 'data' folder.")

    splitter = _get_legal_splitter()
    texts = splitter.split_documents(documents)

    embeddings = _get_embeddings()
    print("📚 Creating embeddings...")

    index_dir = (Path(__file__).resolve().parent / ".faiss_index").resolve()

    try:
        if index_dir.exists():
            # Index was built and saved by this app — safe to load.
            db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded from disk: %s", index_dir)
        else:
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(str(index_dir))
            logger.info("FAISS index built and saved: %s", index_dir)
    except Exception:
        logger.exception("FAISS load failed, rebuilding from documents")
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(str(index_dir))

    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = _get_llm()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": STRICT_LEGAL_PROMPT},
    )
    print("✅ Legal chatbot ready!")
    return chain, db  # Return db so api_server can cache it


# ─── Case Document Analysis ──────────────────────────────────────────────

def _coerce_uploaded_file_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        raise ValueError("No document was uploaded.")
    if isinstance(uploaded_file, bytes):
        return uploaded_file
    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except (OSError, ValueError):
            pass
    if hasattr(uploaded_file, "read"):
        content = uploaded_file.read()
    else:
        content = uploaded_file
    if isinstance(content, str):
        content = content.encode("utf-8")
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("Uploaded document could not be read as bytes.")
    content = bytes(content)
    if not content:
        raise ValueError("Uploaded document is empty.")
    return content


def _clean_extracted_documents(documents: list[Document]) -> list[Document]:
    cleaned = []
    for doc in documents or []:
        text = (getattr(doc, "page_content", "") or "").strip()
        if text:
            doc.page_content = text
            cleaned.append(doc)
    return cleaned


def _uploaded_file_name(uploaded_file) -> str:
    file_name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", None)
    return Path(file_name or "document.pdf").name


def analyze_case_document(uploaded_file):
    """
    Analyze uploaded PDF/TXT. Returns (case_faiss_index, case_summary).

    Optimizations:
    - Single Gemini call for ALL pages (not per-page)
    - 150 DPI (not 300)
    - Gemini output IS the summary (no Ollama re-summarisation)
    """
    print("📄 Analyzing uploaded case document...")

    file_name = _uploaded_file_name(uploaded_file)
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise ValueError("Unsupported document type. Please upload a PDF or TXT file.")

    content = _coerce_uploaded_file_bytes(uploaded_file)
    logger.info("Starting case analysis: file=%s bytes=%d", file_name, len(content))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            print("🤖 Using Gemini Vision for PDF analysis...")
            from pdf2image import convert_from_path

            poppler_path = os.getenv("POPPLER_PATH")

            # 150 DPI: 2× faster, 4× less memory than 300 DPI. Gemini reads fine.
            pages = convert_from_path(tmp_path, dpi=150, poppler_path=poppler_path)
            print(f"  📃 Converted {len(pages)} pages at 150 DPI")

            model = _get_gemini_model()

            # SINGLE Gemini call with ALL pages. Not per-page.
            content_parts = [GEMINI_DOCUMENT_PROMPT] + list(pages)
            print(f"  🚀 Sending {len(pages)} pages to Gemini in ONE call...")
            response = model.generate_content(content_parts)
            extracted_text = response.text

            documents = [
                Document(page_content=extracted_text, metadata={"source": file_name})
            ]
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
            documents = loader.load()

        logger.info("Loaded document: pages=%d file=%s", len(documents), file_name)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not documents:
        raise ValueError("Uploaded document is empty or unreadable.")

    documents = _clean_extracted_documents(documents)
    extracted_chars = sum(len(doc.page_content) for doc in documents)
    logger.info("Extracted text: pages=%d chars=%d file=%s", len(documents), extracted_chars, file_name)

    if not documents or extracted_chars == 0:
        if suffix == ".pdf":
            raise ValueError(
                "No extractable text found in this PDF. "
                "Please upload a text-based PDF/TXT file or run OCR first."
            )
        raise ValueError("Uploaded document contains no readable text.")

    # Chunk for FAISS
    splitter = _get_legal_splitter()
    case_chunks = _clean_extracted_documents(splitter.split_documents(documents))
    print(f"  ✂️ Split into {len(case_chunks)} chunks.")
    logger.info("Chunks=%d file=%s", len(case_chunks), file_name)

    if not case_chunks:
        raise ValueError("Document text could not be split into searchable chunks.")

    # Build FAISS index
    embeddings = _get_embeddings()
    try:
        case_db = FAISS.from_documents(case_chunks, embeddings)
    except IndexError as exc:
        raise ValueError("Document could not be indexed.") from exc

    print("  🗂️ FAISS index created.")

    # Gemini output IS the summary. No Ollama re-summarisation needed.
    case_summary = extracted_text.strip() if suffix == ".pdf" else ""

    # For TXT files, generate a quick summary via Ollama (Gemini wasn't used)
    if suffix == ".txt" and documents:
        full_text = "\n".join(doc.page_content for doc in documents)
        max_chars = int(os.getenv("CASE_SUMMARY_MAX_CHARS", "6000"))
        truncated = full_text[:max_chars]
        llm = _get_llm()
        prompt_text = (
            "You are a legal case analyst specializing in Indian law. "
            "Given the following legal document text, extract:\n"
            "- Parties Involved\n- Key Facts & Timeline\n- Legal Issues / Charges\n"
            "- Evidence Mentioned\n- Relief Sought / Outcome\n\n"
            f"Document Text:\n{truncated}\n\n"
            "Provide extracted details as a structured bullet list."
        )
        try:
            result = llm.invoke(prompt_text)
            case_summary = result.content if hasattr(result, "content") else str(result)
        except Exception:
            logger.exception("TXT summary generation failed")
            case_summary = "Summary could not be generated."

    if not case_summary:
        case_summary = "Summary could not be generated."

    print("  ✅ Case analysis complete.")
    return case_db, case_summary


# ─── Combined RAG Chain ──────────────────────────────────────────────────

class CombinedRAGChain(dict):
    """Adapter for dual-retriever RAG. Matches LangChain invoke API."""

    def invoke(self, inputs):
        question = inputs.get("question", "") if isinstance(inputs, dict) else str(inputs)
        answer = _ask_combined_question(self, question)
        return {"answer": answer}


def build_combined_chain(case_db, law_db=None):
    """
    Build dual-retriever chain. Accepts pre-loaded law_db to avoid disk reload.

    If law_db is None, loads from disk (fallback for backward compat).
    """
    embeddings = _get_embeddings()

    if law_db is None:
        index_dir = (Path(__file__).resolve().parent / ".faiss_index").resolve()
        if index_dir.exists():
            law_db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        else:
            documents = load_legal_docs("data")
            splitter = _get_legal_splitter()
            texts = splitter.split_documents(documents)
            law_db = FAISS.from_documents(texts, embeddings)
            law_db.save_local(str(index_dir))

    legal_retriever = law_db.as_retriever(search_kwargs={"k": 5})
    case_retriever = case_db.as_retriever(search_kwargs={"k": 4})

    return CombinedRAGChain({
        "llm": _get_llm(),
        "legal_retriever": legal_retriever,
        "case_retriever": case_retriever,
    })


def _ask_combined_question(chain, question):
    question = (question or "").strip()
    if not question:
        return "Please ask a legal question about the uploaded document."

    try:
        legal_docs = chain["legal_retriever"].invoke(question)
    except Exception:
        logger.exception("Legal retriever failed")
        legal_docs = []

    try:
        case_docs = chain["case_retriever"].invoke(question)
    except Exception:
        logger.exception("Case retriever failed")
        case_docs = []

    all_docs = legal_docs + case_docs
    if not all_docs:
        return "The available context does not contain enough information to answer this question confidently."

    context = "\n\n".join(
        doc.page_content.strip() for doc in all_docs
        if (getattr(doc, "page_content", "") or "").strip()
    )
    if not context:
        return "The available context does not contain enough readable information to answer this question."

    prompt = COMBINED_QA_PROMPT.format(context=context, question=question)

    try:
        result = chain["llm"].invoke(prompt)
    except Exception:
        logger.exception("Combined LLM invocation failed")
        raise

    if hasattr(result, "content"):
        return result.content
    if isinstance(result, dict):
        return str(result.get("answer") or result.get("content") or result.get("text") or result)
    return str(result)


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    qa_chain, _ = build_legal_chain()
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        target_lang = detect_output_language(query)
        result = qa_chain.invoke({"question": query, "chat_history": []})
        answer = result["answer"]
        translated_answer = translate_answer(answer, target_lang)
        print(f"\nBot ({target_lang}): {translated_answer}")