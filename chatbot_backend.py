from __future__ import annotations

import io
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
except Exception:  # pragma: no cover
    from langchain.prompts import PromptTemplate
try:
    from langchain_classic.chains import ConversationalRetrievalChain
except Exception:  # pragma: no cover
    from langchain.chains import ConversationalRetrievalChain

from langchain.retrievers import EnsembleRetriever


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
        "1️⃣ Use the provided context to answer the question factually and accurately.\n"
        "2️⃣ When the context contains case-specific facts (parties, dates, events), reference them directly.\n"
        "3️⃣ When the context contains legal provisions, cite the specific article, section, or act by name.\n"
        "4️⃣ If the question relates to an uploaded case document, connect the case facts to the relevant law.\n"
        "5️⃣ Prefer Bharatiya Nyaya Sanhita, 2023 (BNS) references over IPC where applicable.\n"
        "6️⃣ If the answer cannot be determined from the provided context, explicitly state: "
        "'The provided context does not contain sufficient information to answer this question confidently.'\n"
        "7️⃣ Do NOT invent or hallucinate any law, section number, or case fact not present in the context.\n"
        "8️⃣ Be precise, lawful, and formal.\n\n"
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
        "1️⃣ Use the given context primarily to answer factually and truthfully.\n"
        "2️⃣ If the question is indirectly related or not explicitly covered in the context, "
        "apply your deep reasoning and general knowledge of INDIAN LAW to answer accurately.\n"
        "3️⃣ Always ensure that your answer strictly pertains to Indian legal systems, acts, amendments, "
        "articles, and judicial practices.\n"
        "4️⃣ Never discuss or compare with foreign countries or laws unless it helps clarify Indian context.\n"
        "5️⃣ If absolutely no relevant information is available, respond with:\n"
        "'The provided context does not contain this information, but under Indian law, it can be interpreted as follows...' "
        "and then give a reasoned Indian legal explanation if possible.\n"
        "6️⃣ Be precise, lawful, and formal — avoid speculation or personal opinions.\n"
        "7️⃣ Penal law references: Prefer Bharatiya Nyaya Sanhita, 2023 (BNS) section references over IPC. "
        "If the user asks about an IPC section, answer using the corresponding BNS provision when you are confident. "
        "If you are not confident about the exact IPC→BNS section number mapping, do NOT guess; instead explain that IPC has been replaced by BNS and provide the relevant offence/topic under BNS in a way the user can verify.\n\n"
        "Important stylistic rule: Do NOT state or imply whether any part of the answer 'comes from' the provided context "
        "or 'comes from' the assistant's internal knowledge. Present conclusions, reasoning and citations seamlessly — "
        "do not include meta-statements about source or provenance of individual sentences.\n\n"
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
    file_name = getattr(uploaded_file, "name", "document.pdf")
    suffix = Path(file_name).suffix.lower()

    # --- Save to temp file (PyPDFLoader needs a path) ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file)
        tmp_path = tmp.name

    try:
        # --- Load ---
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        documents = loader.load()
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not documents:
        raise ValueError("Uploaded document is empty or unreadable.")

    # --- Chunk ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    case_chunks = splitter.split_documents(documents)
    print(f"  ✂️  Split into {len(case_chunks)} chunks.")

    # --- Build in-memory FAISS index ---
    embeddings = _get_embeddings()
    case_db = FAISS.from_documents(case_chunks, embeddings)
    print("  🗂️  In-memory FAISS index created.")

    # --- Generate case summary via LLM ---
    full_text = "\n".join(doc.page_content for doc in documents)
    # Truncate to avoid exceeding context window
    max_chars = int(os.getenv("CASE_SUMMARY_MAX_CHARS", "6000"))
    truncated_text = full_text[:max_chars]
    if len(full_text) > max_chars:
        truncated_text += "\n\n[... document truncated for summary extraction ...]"

    llm = _get_llm()
    summary_input = CASE_ANALYSIS_PROMPT.format(text=truncated_text)
    case_summary = llm.predict(summary_input) if hasattr(llm, "predict") else llm.invoke(summary_input)
    # Handle AIMessage objects
    if hasattr(case_summary, "content"):
        case_summary = case_summary.content

    print("  ✅ Case summary generated.")
    return case_db, case_summary


def build_combined_chain(case_db):
    """
    Build a ConversationalRetrievalChain that retrieves from BOTH the uploaded
    case document (case_db) and the persistent legal knowledge base (.faiss_index).

    Uses EnsembleRetriever with Reciprocal Rank Fusion to merge results.

    Parameters
    ----------
    case_db : FAISS
        The in-memory FAISS index built from the uploaded case document.

    Returns
    -------
    ConversationalRetrievalChain
    """
    print("🔗 Building combined (dual-RAG) retrieval chain...")

    embeddings = _get_embeddings()

    # --- Load existing legal FAISS index ---
    index_dir = (Path(__file__).resolve().parent / ".faiss_index").resolve()
    if index_dir.exists():
        law_db = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
    else:
        # Rebuild from source documents if index is missing
        documents = load_legal_docs("data")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = splitter.split_documents(documents)
        law_db = FAISS.from_documents(texts, embeddings)
        law_db.save_local(str(index_dir))

    # --- Create retrievers ---
    legal_retriever = law_db.as_retriever(search_kwargs={"k": 5})
    case_retriever = case_db.as_retriever(search_kwargs={"k": 3})

    # --- Ensemble retriever (Reciprocal Rank Fusion) ---
    combined_retriever = EnsembleRetriever(
        retrievers=[legal_retriever, case_retriever],
        weights=[0.5, 0.5],
    )

    # --- Build chain ---
    llm = _get_llm()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=combined_retriever,
        combine_docs_chain_kwargs={"prompt": COMBINED_QA_PROMPT},
    )

    print("✅ Combined dual-RAG chain is ready!")
    return chain


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
