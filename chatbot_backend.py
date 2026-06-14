from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from pathlib import Path

try:
    from langchain_ollama import ChatOllama  # preferred (newer LangChain)
except Exception:  # pragma: no cover
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover
    ChatOpenAI = None
    OpenAIEmbeddings = None

from langchain_community.llms import Ollama  # fallback (older LangChain)
from langchain_community.document_loaders import TextLoader, PyPDFLoader
try:
    # LangChain 1.x+ (splitters live in a separate package)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    # Older LangChain
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





# ─── Prompt Templates for Case Document Analysis ───────────────────────────

CASE_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a legal case analyst specializing in Indian law. "
        "Given the following legal document text, extract and list the following details:\n"
        "- **Parties Involved**: Names and roles (plaintiff, defendant, petitioner, respondent, etc.)\n"
        "- **Key Facts & Timeline**: Important events in chronological order\n"
        "- **Legal Issues / Charges**: Laws invoked, sections cited, or legal questions raised\n"
        "- **Evidence Mentioned**: Documents, witnesses, or exhibits referenced\n"
        "- **Relief Sought / Outcome**: What was asked for or decided\n\n"
        "Document Text:\n{text}\n\n"
        "Provide the extracted details as a well-structured bullet list. "
        "If any category has no information, state 'Not mentioned in the document.'"
    )
)

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


# 🧩 Detect output language from query (like "in Hindi" or "in Telugu")
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
            "urdu": "ur"
        }
        return lang_map.get(lang_word, "en")
    return "en"  # default → English


# 🌐 Translate English answer → target language using Deep Translator
def translate_answer(answer, target_lang="en"):
    if target_lang == "en":
        return answer
    try:
        # Optional dependency; also requires internet access.
        from deep_translator import GoogleTranslator  # type: ignore

        translated = GoogleTranslator(source="en", target=target_lang).translate(answer)
        print(f"🌍 Translated answer → {target_lang}: {translated}")
        return translated
    except Exception as e:
        print("⚠️ Translation error:", e)
        return answer


# ─── Helper: Get Embeddings ───────────────────────────────────────────────

def _get_embeddings():
    """Return the embedding model instance based on the configured provider."""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if llm_provider == "openai":
        if OpenAIEmbeddings is None:
            raise ImportError("LLM_PROVIDER=openai requires 'langchain-openai' to be installed")
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
        )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# ─── Helper: Get LLM ─────────────────────────────────────────────────────

def _get_llm():
    """Return the LLM instance based on the configured provider."""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if llm_provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("LLM_PROVIDER=openai requires 'langchain-openai' to be installed")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        )
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    ollama_base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
    ollama_kwargs = {}
    if ollama_base_url:
        ollama_kwargs["base_url"] = ollama_base_url
    if ChatOllama is not None:
        return ChatOllama(model=model, num_ctx=num_ctx, **ollama_kwargs)
    return Ollama(model=model, num_ctx=num_ctx, **ollama_kwargs)


# 🧩 Step 1: Load both TXT and PDF legal documents (Constitution, Acts, etc.)
def load_legal_docs(folder_path: str | os.PathLike[str] = "data"):
    """Load all .txt and .pdf files from the given folder."""
    folder = Path(folder_path)
    if not folder.is_absolute():
        folder = (Path(__file__).resolve().parent / folder).resolve()

    if not folder.exists():
        raise FileNotFoundError(
            f"Data folder not found: {folder}. Create it and add .txt/.pdf files."
        )

    documents = []
    for file_path in sorted(folder.glob("*")):
        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

    return documents


# 🧠 Step 2: Build the retrieval-based chatbot chain (Multilingual + Context-only)
def build_legal_chain():
    print("🔍 Loading and preparing legal documents...")
    documents = load_legal_docs("data")

    if not documents:
        raise ValueError("⚠️ No legal documents found in the 'data' folder. Please add .txt or .pdf files.")

    # ✅ Better chunking for legal articles
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_documents(documents)

    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    print("📚 Creating embeddings...")
    if llm_provider == "openai":
        if OpenAIEmbeddings is None:
            raise ImportError("LLM_PROVIDER=openai requires 'langchain-openai' to be installed")

        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
        )
    else:
        # ✅ Use multilingual embeddings for Hindi, Telugu, etc.
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    # Cache the vector index locally to avoid re-embedding on every restart.
    # NOTE: Loading a saved FAISS index may require deserializing a pickle file.
    # For safety, this is disabled by default for hosted APIs.
    index_dir = (Path(__file__).resolve().parent / ".faiss_index").resolve()
    allow_pickle_load = os.getenv("FAISS_ALLOW_DANGEROUS_DESERIALIZATION", "0") == "1"

    try:
        if index_dir.exists() and allow_pickle_load:
            db = FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(str(index_dir))
    except Exception:
        # If the cache is corrupted or incompatible, rebuild it.
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(str(index_dir))

    retriever = db.as_retriever(search_kwargs={"k": 5})

    if llm_provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("LLM_PROVIDER=openai requires 'langchain-openai' to be installed")

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        )
    else:
        # ✅ Ollama model (keep it lightweight by default)
        # IMPORTANT: default to a general chat/instruct model (not a coding model),
        # otherwise answers may sound like a programming assistant.
        # Override with env var: set OLLAMA_MODEL=llama3.2:3b (or similar)
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

        # For remote Ollama (e.g., when UI runs on Streamlit Cloud), set one of:
        # - OLLAMA_BASE_URL=http://<server>:11434
        # - OLLAMA_HOST=http://<server>:11434
        ollama_base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
        ollama_kwargs = {}
        if ollama_base_url:
            ollama_kwargs["base_url"] = ollama_base_url

        if ChatOllama is not None:
            llm = ChatOllama(model=model, num_ctx=num_ctx, **ollama_kwargs)
        else:
            llm = Ollama(model=model, num_ctx=num_ctx, **ollama_kwargs)

    # If you have enough VRAM, you can try: llm = Ollama(model="llama3:instruct", num_ctx=2048)

    # ⚖️ Strict prompt to avoid irrelevant (US) answers
       # ⚖️ Smarter but India-focused legal reasoning prompt
    strict_prompt = PromptTemplate(
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

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": strict_prompt}
    )

    print("✅ Legal multilingual chatbot is ready!")
    return chain


# ─── Case Document Analysis ────────────────────────────────────────────────

def _coerce_uploaded_file_bytes(uploaded_file) -> bytes:
    """Read uploaded bytes without assuming a specific web framework object."""
    if uploaded_file is None:
        raise ValueError("No document was uploaded.")

    if isinstance(uploaded_file, bytes):
        return uploaded_file

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except (OSError, ValueError):
            logger.debug("Uploaded file object could not be rewound before reading.", exc_info=True)

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
    """Keep only pages/chunks with extractable text."""
    cleaned = []
    for doc in documents or []:
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        doc.page_content = text
        cleaned.append(doc)
    return cleaned


def _uploaded_file_name(uploaded_file) -> str:
    file_name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", None)
    return Path(file_name or "document.pdf").name


def analyze_case_document(uploaded_file):
    """
    Load an uploaded PDF or TXT file, split it into chunks, build an in-memory
    FAISS index, and generate a structured case summary using the LLM.

    Parameters
    ----------
    uploaded_file : file-like object (BytesIO / Streamlit UploadedFile)
        The uploaded legal document.

    Returns
    -------
    tuple[FAISS, str]
        (case_faiss_index, case_summary_string)

    Raises
    ------
    ValueError
        If the document is empty or unreadable.
    """
    print("📄 Analyzing uploaded case document...")

    # --- Determine file type from name attribute ---
    file_name = _uploaded_file_name(uploaded_file)
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise ValueError("Unsupported document type. Please upload a PDF or TXT file.")

    content = _coerce_uploaded_file_bytes(uploaded_file)
    logger.info(
        "Starting case document analysis: file_name=%s suffix=%s bytes=%d",
        file_name,
        suffix,
        len(content),
    )

    # --- Save to temp file (PyPDFLoader needs a path) ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # --- Load ---
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        documents = loader.load()
        logger.info("Loaded uploaded document pages=%d file_name=%s", len(documents), file_name)
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not documents:
        logger.warning("Document loader returned zero pages: file_name=%s", file_name)
        raise ValueError("Uploaded document is empty or unreadable.")

    documents = _clean_extracted_documents(documents)
    extracted_chars = sum(len(doc.page_content) for doc in documents)
    logger.info(
        "Extracted text from uploaded document: text_pages=%d chars=%d file_name=%s",
        len(documents),
        extracted_chars,
        file_name,
    )

    if not documents or extracted_chars == 0:
        logger.warning(
            "No extractable text found in uploaded document: file_name=%s suffix=%s",
            file_name,
            suffix,
        )
        if suffix == ".pdf":
            raise ValueError(
                "No extractable text was found in this PDF. It appears to be scanned or image-only. "
                "Please upload a text-based PDF/TXT file or run OCR on the PDF first."
            )
        raise ValueError("Uploaded document contains no readable text.")

    # --- Chunk ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    case_chunks = _clean_extracted_documents(splitter.split_documents(documents))
    print(f"  ✂️  Split into {len(case_chunks)} chunks.")
    logger.info("Chunked uploaded document: chunks=%d file_name=%s", len(case_chunks), file_name)

    if not case_chunks:
        logger.warning("Text splitter returned zero chunks: file_name=%s", file_name)
        raise ValueError("Uploaded document text could not be split into searchable chunks.")

    # --- Build in-memory FAISS index ---
    embeddings = _get_embeddings()
    try:
        case_db = FAISS.from_documents(case_chunks, embeddings)
    except IndexError as exc:
        logger.exception(
            "FAISS index creation failed because no embeddings were produced: file_name=%s chunks=%d",
            file_name,
            len(case_chunks),
        )
        raise ValueError("Uploaded document could not be indexed because no text embeddings were produced.") from exc
    print("  🗂️  In-memory FAISS index created.")
    logger.info(
        "In-memory FAISS index created: vectors=%s file_name=%s",
        getattr(getattr(case_db, "index", None), "ntotal", "unknown"),
        file_name,
    )

    # --- Generate case summary via LLM ---
    full_text = "\n".join(doc.page_content for doc in documents)
    # Truncate to avoid exceeding context window
    max_chars = int(os.getenv("CASE_SUMMARY_MAX_CHARS", "6000"))
    truncated_text = full_text[:max_chars]
    if len(full_text) > max_chars:
        truncated_text += "\n\n[... document truncated for summary extraction ...]"

    llm = _get_llm()
    summary_input = CASE_ANALYSIS_PROMPT.format(text=truncated_text)
    try:
        case_summary = llm.predict(summary_input) if hasattr(llm, "predict") else llm.invoke(summary_input)
    except Exception:
        logger.exception("Case summary generation failed: file_name=%s", file_name)
        raise
    # Handle AIMessage objects
    if hasattr(case_summary, "content"):
        case_summary = case_summary.content
    if isinstance(case_summary, dict):
        case_summary = case_summary.get("content") or case_summary.get("text") or str(case_summary)
    if case_summary is None:
        logger.warning("LLM returned no case summary content: file_name=%s", file_name)
        case_summary = "Case summary could not be generated from the extracted text."
    case_summary = str(case_summary).strip()
    if not case_summary:
        logger.warning("LLM returned an empty case summary: file_name=%s", file_name)
        case_summary = "Case summary could not be generated from the extracted text."

    print("  ✅ Case summary generated.")
    return case_db, case_summary


class CombinedRAGChain(dict):
    """Small adapter so the manual dual-RAG chain still matches LangChain's invoke API."""

    def invoke(self, inputs):
        if isinstance(inputs, dict):
            question = inputs.get("question", "")
        else:
            question = str(inputs)
        answer = ask_combined_question(self, question)
        return {"answer": answer}


def build_combined_chain(case_db):
    embeddings = _get_embeddings()

    index_dir = (Path(__file__).resolve().parent / ".faiss_index").resolve()

    if index_dir.exists():
        law_db = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        documents = load_legal_docs("data")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        texts = splitter.split_documents(documents)

        law_db = FAISS.from_documents(texts, embeddings)
        law_db.save_local(str(index_dir))

    legal_retriever = law_db.as_retriever(search_kwargs={"k": 5})
    case_retriever = case_db.as_retriever(search_kwargs={"k": 3})

    return CombinedRAGChain({
        "llm": _get_llm(),
        "legal_retriever": legal_retriever,
        "case_retriever": case_retriever,
    })

def ask_combined_question(chain, question):
    question = (question or "").strip()
    if not question:
        return "Please ask a legal question about the uploaded document."

    try:
        legal_docs = chain["legal_retriever"].invoke(question)
    except Exception:
        logger.exception("Legal retriever failed during combined question.")
        legal_docs = []

    try:
        case_docs = chain["case_retriever"].invoke(question)
    except Exception:
        logger.exception("Case retriever failed during combined question.")
        case_docs = []

    all_docs = legal_docs + case_docs
    if not all_docs:
        logger.warning("Combined retrievers returned no documents for question.")
        return "The available context does not contain enough information to answer this question confidently."

    context = "\n\n".join(
        [(getattr(doc, "page_content", "") or "").strip() for doc in all_docs if (getattr(doc, "page_content", "") or "").strip()]
    )
    if not context:
        logger.warning("Combined retrievers returned documents without text content.")
        return "The available context does not contain enough readable information to answer this question confidently."

    prompt = COMBINED_QA_PROMPT.format(
        context=context,
        question=question
    )

    try:
        result = chain["llm"].invoke(prompt)
    except Exception:
        logger.exception("Combined LLM invocation failed.")
        raise

    if hasattr(result, "content"):
        return result.content
    if isinstance(result, dict):
        return str(result.get("answer") or result.get("content") or result.get("text") or result)

    return str(result)

# 💬 Step 3: Example of usage
if __name__ == "__main__":
    qa_chain = build_legal_chain()

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Detect target language (e.g. "in Telugu", "in Hindi")
        target_lang = detect_output_language(query)
        print(f"🈯 Detected target language: {target_lang}")

        result = qa_chain.invoke({"question": query, "chat_history": []})
        answer = result["answer"]

        # Translate answer to target language
        translated_answer = translate_answer(answer, target_lang)
        print(f"\nBot ({target_lang}): {translated_answer}")
