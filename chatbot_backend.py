from __future__ import annotations

import os
import re
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

# 🧩 Detect output language from query (like “in Hindi” or “in Telugu”)
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
