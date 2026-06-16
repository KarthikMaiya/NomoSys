"""NomoSys Legal Chatbot Backend — v5.0 Demo-Day Edition.

Key changes vs v4:
- Prompts fully rewritten: lawyer-like, non-robotic, BNS-first, no retrieval language
- Complaint generator added (generate_legal_complaint)
- Legal insights extraction (_extract_legal_insights)
- Case timeline extraction (_extract_case_timeline)
- Conversation history persistence helpers
- Follow-up questions and risk level in every response format
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

from langchain_community.document_loaders import TextLoader, PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except Exception:
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document

try:
    from langchain_classic.chains import ConversationalRetrievalChain
except Exception:
    from langchain.chains import ConversationalRetrievalChain

logger = logging.getLogger(__name__)

# ─── Singletons ──────────────────────────────────────────────────────────────

_cached_embeddings: Optional[HuggingFaceEmbeddings] = None
_cached_llm = None
_cached_gemini_model = None
_cached_legal_chain = None
_cached_law_db = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _cached_embeddings
    if _cached_embeddings is not None:
        logger.info("EMBEDDINGS REUSED")
        return _cached_embeddings
    logger.info("EMBEDDINGS LOADED")
    _cached_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("Embedding model loaded (singleton)")
    return _cached_embeddings


def _get_llm():
    global _cached_llm
    if _cached_llm is not None:
        logger.info("LLM REUSED")
        return _cached_llm
    if ChatOllama is None:
        raise RuntimeError("langchain_ollama not installed. Run: pip install langchain-ollama")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    kwargs = {}
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
    if base_url:
        kwargs["base_url"] = base_url
    _cached_llm = ChatOllama(model=model, num_ctx=num_ctx, **kwargs)
    logger.info("LLM LOADED: model=%s", model)
    return _cached_llm


def _get_gemini_model():
    global _cached_gemini_model
    if _cached_gemini_model is not None:
        return _cached_gemini_model
    _cached_gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    logger.info("Gemini model loaded (singleton)")
    return _cached_gemini_model


# ─── Prompts ─────────────────────────────────────────────────────────────────

GEMINI_FIR_PROMPT = """\
You are a senior Indian legal analyst with 20 years of experience in criminal and civil law.
Analyze this legal document and extract ALL information with precision.
Return ONLY valid JSON — no markdown, no backticks, no preamble, no explanation.

JSON schema (fill every field; use null if not found):
{
  "document_type": "FIR | Judgment | Contract | Agreement | Notice | Other",
  "case_number": "string or null",
  "police_station": "string or null",
  "court": "string or null",
  "date_of_filing": "string or null",
  "complainant": {
    "name": "string or null",
    "address": "string or null",
    "contact": "string or null"
  },
  "victim": {
    "name": "string or null",
    "relation_to_complainant": "string or null"
  },
  "accused": [
    {"name": "string", "address": "string or null", "description": "string or null"}
  ],
  "incident": {
    "date": "string or null",
    "time": "string or null",
    "location": "string or null",
    "description": "string — detailed factual summary of what happened"
  },
  "applicable_sections": [
    {"act": "string", "section": "string", "description": "string"}
  ],
  "evidence_mentioned": ["string"],
  "witnesses": ["string"],
  "relief_sought": "string or null",
  "important_dates": [{"event": "string", "date": "string"}],
  "case_strength": {
    "rating": "Strong | Moderate | Weak",
    "reasons": ["string"],
    "risks": ["string"]
  },
  "legal_insights": {
    "primary_offence": "string or null",
    "strongest_evidence": "string or null",
    "missing_evidence": ["string"],
    "investigation_gaps": ["string"],
    "potential_risks": ["string"]
  },
  "legal_summary": "3-4 sentence plain English summary of the case for a layperson",
  "suggested_questions": [
    "string", "string", "string", "string", "string"
  ]
}

Analyze the document now. Return only the JSON object."""

GEMINI_TXT_PROMPT = """\
You are a senior Indian legal analyst.
Analyze this legal document text and extract ALL information with precision.
Return ONLY valid JSON — no markdown, no backticks, no preamble.

JSON schema (fill every field; use null if not found):
{
  "document_type": "string",
  "case_number": "string or null",
  "parties": [{"role": "string", "name": "string", "details": "string or null"}],
  "incident": {
    "date": "string or null",
    "description": "string"
  },
  "applicable_sections": [
    {"act": "string", "section": "string", "description": "string"}
  ],
  "key_facts": ["string"],
  "evidence_mentioned": ["string"],
  "relief_sought": "string or null",
  "important_dates": [{"event": "string", "date": "string"}],
  "case_strength": {
    "rating": "Strong | Moderate | Weak",
    "reasons": ["string"],
    "risks": ["string"]
  },
  "legal_insights": {
    "primary_offence": "string or null",
    "strongest_evidence": "string or null",
    "missing_evidence": ["string"],
    "investigation_gaps": ["string"],
    "potential_risks": ["string"]
  },
  "legal_summary": "3-4 sentence plain English summary",
  "suggested_questions": ["string", "string", "string", "string", "string"]
}

Return only the JSON object."""

CASE_QA_PROMPT_TEMPLATE = """\
You are NomoSys, a highly experienced Indian legal assistant specialising in BNS 2023, BNSS, BSA, Indian Constitution, and civil law.

A client has uploaded their legal document. Here are the extracted case facts:

CASE FACTS:
{case_context}

RELEVANT LEGAL PROVISIONS FROM INDIAN LAW DATABASE:
{law_context}

CLIENT QUESTION: {question}

INSTRUCTIONS:
- Respond as an experienced Indian advocate speaking directly to their client
- Reference parties, dates, and locations by name from the case facts above
- NEVER say "the context states", "based on retrieved information", "the document mentions", "according to the provided context", or any similar phrase
- Prefer BNS 2023 / BNSS / BSA over IPC / CrPC / Indian Evidence Act unless the matter predates those laws or the client specifically asks
- Do not invent section numbers not present in the legal provisions
- If a specific detail is missing, say "I would need to see [document/detail] to advise on this"
- Be empathetic, direct, and practical

Respond in EXACTLY this format:

⚖️ Legal Assessment
[3-5 sentences of plain-English legal analysis specific to this case. Name the parties and facts directly.]

📚 Applicable Laws
[Each relevant provision on its own line: "• BNS Section X — one-line explanation of how it applies here"]

📋 Recommended Actions
1.
2.
3.

📑 Evidence & Documents Required
• 
• 

🚨 Risk Level: [Low / Moderate / High]
[1-2 sentences on the specific legal risks in this matter]

📚 Sources
[List each Act/Section cited above as bullet points]

💡 Follow-Up Questions
• 
• 
• """

LAW_QA_PROMPT_TEMPLATE = """\
You are NomoSys, a highly experienced Indian legal assistant with deep expertise in BNS 2023, BNSS, BSA, Indian constitutional law, and civil law.

RELEVANT LEGAL PROVISIONS:
{context}

CLIENT QUESTION: {question}

INSTRUCTIONS:
- Respond like an experienced Indian advocate explaining the law to a client in plain language
- NEVER say "the context states", "based on retrieved information", "according to the provided context", or any similar AI system language
- Prefer BNS 2023 / BNSS / BSA. Only cite IPC / CrPC if the matter predates the new codes or the client specifically asks
- For common citizen issues (lost documents, online harassment, consumer complaints, tenancy), lead with practical steps first
- Do not cite a section merely to appear more legal — only cite when materially relevant
- If a question falls outside available provisions, say so honestly and give practical guidance

Respond in EXACTLY this format:

⚖️ Legal Assessment
[Clear explanation of the legal position under Indian law. Plain English. 3-5 sentences.]

📚 Applicable Laws
[Each relevant provision on its own line: "• BNS Section X — one-line practical explanation"]

📋 Recommended Actions
1.
2.
3.

📑 Evidence & Documents Required
• 
• 

🚨 Risk Level: [Low / Moderate / High]
[Brief explanation of legal risk in this situation]

📚 Sources
[List each Act/Section cited above as bullet points]

💡 Follow-Up Questions
• 
• 
• """

STRICT_LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=LAW_QA_PROMPT_TEMPLATE,
)

COMPLAINT_GENERATION_PROMPT = """\
You are a senior Indian legal drafting specialist. Generate a professional legal complaint letter for filing with the relevant authority.

COMPLAINT TYPE: {complaint_type}
COMPLAINANT NAME: {complainant_name}
COMPLAINANT ADDRESS: {complainant_address}
INCIDENT DESCRIPTION: {incident_description}
DATE OF INCIDENT: {incident_date}
ACCUSED / RESPONDENT: {accused_name}
ADDITIONAL DETAILS: {additional_details}

Generate a complete, formal complaint letter following Indian legal conventions.
Address it to the appropriate authority (Station House Officer for criminal matters, relevant Consumer Forum, etc.).
Include: proper salutation, subject line, factual narration in chronological order, applicable legal provisions under BNS 2023 / BNSS / BSA / Consumer Protection Act as relevant, relief sought, and closing.
Use formal legal language but keep it clear and precise.
Do NOT include placeholder brackets — fill in all details from the information provided.
If a detail is not provided, omit that part gracefully.

Output the complete letter only, starting with "To," and ending with the signature block."""


# ─── Language Detection & Translation ────────────────────────────────────────

def detect_output_language(query: str) -> str:
    lang_map = {
        "english": "en", "hindi": "hi", "kannada": "kn", "tamil": "ta",
        "telugu": "te", "malayalam": "ml", "marathi": "mr", "bengali": "bn",
        "gujarati": "gu", "urdu": "ur", "punjabi": "pa", "arabic": "ar",
    }
    match = re.search(r"\bin\s+([a-z]+)\b", query or "", re.IGNORECASE)
    if match:
        lang_word = match.group(1).lower()
        if lang_word in lang_map:
            return lang_map[lang_word]

    script_patterns = [
        (r"[\u0900-\u097F]", "hi"),
        (r"[\u0C00-\u0C7F]", "te"),
        (r"[\u0B80-\u0BFF]", "ta"),
        (r"[\u0C80-\u0CFF]", "kn"),
        (r"[\u0D00-\u0D7F]", "ml"),
        (r"[\u0980-\u09FF]", "bn"),
        (r"[\u0A00-\u0A7F]", "pa"),
        (r"[\u0600-\u06FF]", "ur"),
        (r"[\u0A80-\u0AFF]", "gu"),
    ]
    for pattern, code in script_patterns:
        if re.search(pattern, query or ""):
            return code
    return "en"


def translate_answer(answer: str, target_lang: str = "en") -> str:
    if target_lang == "en":
        return answer
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=target_lang).translate(answer)
    except Exception as e:
        logger.warning("Translation error: %s", e)
        return answer


# ─── Legal Knowledge Base ─────────────────────────────────────────────────────

def load_legal_docs(folder_path: str | os.PathLike[str] = "data"):
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
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "Article ", "Section ", ". ", " "],
    )


def build_legal_chain():
    """Build RAG chain for legal KB. Cached as singleton — runs exactly once."""
    global _cached_legal_chain, _cached_law_db

    if _cached_legal_chain is not None:
        logger.info("CHAIN REUSED")
        return _cached_legal_chain, _cached_law_db

    logger.info("BUILD LEGAL CHAIN CALLED")
    print("Loading legal knowledge base...")
    documents = load_legal_docs("data")
    if not documents:
        raise ValueError("No legal documents found in the 'data' folder.")

    splitter = _get_legal_splitter()
    texts = splitter.split_documents(documents)
    embeddings = _get_embeddings()

    index_dir = (Path(__file__).resolve().parent / ".faiss_index").resolve()
    try:
        if index_dir.exists():
            db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS LOADED from disk: %s", index_dir)
        else:
            print("Building FAISS index (first run — will cache for future)...")
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(str(index_dir))
            logger.info("FAISS LOADED (built and saved): %s", index_dir)
    except Exception:
        logger.exception("FAISS load failed, rebuilding")
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(str(index_dir))

    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = _get_llm()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": STRICT_LEGAL_PROMPT},
    )

    _cached_legal_chain = chain
    _cached_law_db = db

    print("Legal knowledge base ready!")
    logger.info("READY")
    return _cached_legal_chain, _cached_law_db


# ─── Case Document Parsing ────────────────────────────────────────────────────

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
    content = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    if isinstance(content, str):
        content = content.encode("utf-8")
    content = bytes(content)
    if not content:
        raise ValueError("Uploaded document is empty.")
    return content


def _uploaded_file_name(uploaded_file) -> str:
    name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", None)
    return Path(name or "document.pdf").name


def _parse_gemini_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Gemini returned non-JSON response; using raw text as summary")
    return {"legal_summary": raw, "_raw": True}


def _extract_case_timeline(case_data: dict) -> list[dict]:
    """Extract chronological timeline from case data."""
    events = []

    filing_date = case_data.get("date_of_filing")
    doc_type = case_data.get("document_type", "Document")
    if filing_date:
        events.append({"date": filing_date, "event": f"{doc_type} Filed"})

    incident = case_data.get("incident", {})
    if incident and incident.get("date"):
        events.append({"date": incident["date"], "event": "Incident Occurred"})

    for entry in case_data.get("important_dates", []) or []:
        if entry.get("date") and entry.get("event"):
            events.append({"date": entry["date"], "event": entry["event"]})

    # Deduplicate
    seen = set()
    unique = []
    for e in events:
        key = (e["date"], e["event"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    try:
        unique.sort(key=lambda x: x.get("date", ""))
    except Exception:
        pass

    return unique


def _format_case_summary(case_data: dict) -> str:
    """Convert structured JSON into rich markdown for the UI."""
    if case_data.get("_raw"):
        return case_data.get("legal_summary", "Summary unavailable.")

    lines = []

    doc_type = case_data.get("document_type", "Legal Document")
    case_num = case_data.get("case_number") or case_data.get("fir_number")
    lines.append(f"## 📄 {doc_type}" + (f" — {case_num}" if case_num else ""))

    summary = case_data.get("legal_summary")
    if summary:
        lines.append(f"\n{summary}")

    # Parties
    complainant = case_data.get("complainant", {})
    victim = case_data.get("victim", {})
    accused_list = case_data.get("accused", [])
    parties = case_data.get("parties", [])

    if complainant or victim or accused_list or parties:
        lines.append("\n### 👥 Parties Involved")
        if complainant and complainant.get("name"):
            lines.append(f"- **Complainant:** {complainant['name']}")
        if victim and victim.get("name"):
            lines.append(f"- **Victim:** {victim['name']}")
        for acc in accused_list or []:
            name = acc.get("name", "Unknown")
            desc = acc.get("description", "")
            lines.append(f"- **Accused:** {name}" + (f" ({desc})" if desc else ""))
        for p in parties or []:
            lines.append(f"- **{p.get('role', 'Party')}:** {p.get('name', '')}")

    ps = case_data.get("police_station")
    court = case_data.get("court")
    if ps:
        lines.append(f"\n🏢 **Police Station:** {ps}")
    if court:
        lines.append(f"⚖️ **Court:** {court}")

    # Incident
    incident = case_data.get("incident", {})
    if incident:
        lines.append("\n### 📌 Incident Details")
        if incident.get("date"):
            lines.append(f"- **Date:** {incident['date']}")
        if incident.get("time"):
            lines.append(f"- **Time:** {incident['time']}")
        if incident.get("location"):
            lines.append(f"- **Location:** {incident['location']}")
        if incident.get("description"):
            lines.append(f"\n{incident['description']}")

    # Applicable sections
    sections = case_data.get("applicable_sections", [])
    if sections:
        lines.append("\n### ⚖️ Charges & Applicable Law")
        for s in sections:
            act = s.get("act", "")
            sec = s.get("section", "")
            desc = s.get("description", "")
            lines.append(f"- **{act} {sec}** — {desc}")

    # Evidence
    evidence = case_data.get("evidence_mentioned", [])
    if evidence:
        lines.append("\n### 📎 Evidence on Record")
        for e in evidence:
            lines.append(f"- {e}")

    # Legal Insights (NEW)
    insights = case_data.get("legal_insights", {})
    if insights:
        lines.append("\n### 🔍 Key Legal Insights")
        if insights.get("primary_offence"):
            lines.append(f"- **Primary Offence:** {insights['primary_offence']}")
        if insights.get("strongest_evidence"):
            lines.append(f"- **Strongest Evidence:** {insights['strongest_evidence']}")
        for item in insights.get("missing_evidence", []):
            lines.append(f"- ⚠️ **Missing Evidence:** {item}")
        for item in insights.get("investigation_gaps", []):
            lines.append(f"- 🔎 **Investigation Gap:** {item}")
        for item in insights.get("potential_risks", []):
            lines.append(f"- 🚨 **Potential Risk:** {item}")

    # Case Timeline (NEW)
    timeline = _extract_case_timeline(case_data)
    if timeline:
        lines.append("\n### 📅 Case Timeline")
        for entry in timeline:
            lines.append(f"- **{entry['date']}** — {entry['event']}")

    # Case strength
    strength = case_data.get("case_strength", {})
    if strength:
        rating = strength.get("rating", "—")
        emoji = {"Strong": "🟢", "Moderate": "🟡", "Weak": "🔴"}.get(rating, "⚪")
        lines.append(f"\n### {emoji} Case Strength: {rating}")
        for r in strength.get("reasons", []):
            lines.append(f"- ✅ {r}")
        for r in strength.get("risks", []):
            lines.append(f"- ⚠️ {r}")

    # Suggested questions
    questions = case_data.get("suggested_questions", [])
    if questions:
        lines.append("\n### 💡 Questions to Ask Your Advocate")
        for i, q in enumerate(questions[:5], 1):
            lines.append(f"{i}. {q}")

    return "\n".join(lines)


def analyze_case_document(uploaded_file) -> tuple[dict, str]:
    """
    Analyze uploaded PDF/TXT.

    Returns:
        (case_data_dict, formatted_summary_markdown)
    """
    print("Analyzing uploaded case document...")

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
        gemini = _get_gemini_model()

        if suffix == ".pdf":
            from pdf2image import convert_from_path
            poppler_path = os.getenv("POPPLER_PATH")
            pages = convert_from_path(tmp_path, dpi=150, poppler_path=poppler_path)
            print(f"  Converted {len(pages)} pages at 150 DPI")
            print(f"  Sending {len(pages)} pages to Gemini...")
            response = gemini.generate_content([GEMINI_FIR_PROMPT] + list(pages))
            raw_response = response.text
        else:
            with open(tmp_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            if not text_content.strip():
                raise ValueError("Uploaded TXT document is empty.")
            print("  Sending document text to Gemini...")
            response = gemini.generate_content(GEMINI_TXT_PROMPT + "\n\nDOCUMENT TEXT:\n" + text_content)
            raw_response = response.text
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not raw_response or not raw_response.strip():
        raise ValueError("Gemini returned an empty response. Please try again.")

    print("  Parsing Gemini structured output...")
    case_data = _parse_gemini_json(raw_response)
    case_data["_source_file"] = file_name
    case_data["_raw_gemini"] = raw_response

    summary_md = _format_case_summary(case_data)
    if not summary_md:
        summary_md = "Document analyzed. Ask questions below."

    print("  Case analysis complete.")
    logger.info("Case analysis complete: file=%s", file_name)
    return case_data, summary_md


# ─── Complaint Generator ──────────────────────────────────────────────────────

def generate_legal_complaint(
    complaint_type: str,
    complainant_name: str,
    complainant_address: str,
    incident_description: str,
    incident_date: str = "",
    accused_name: str = "",
    additional_details: str = "",
) -> str:
    """Generate a formal Indian legal complaint letter via Gemini."""
    gemini = _get_gemini_model()
    prompt = COMPLAINT_GENERATION_PROMPT.format(
        complaint_type=complaint_type or "General Complaint",
        complainant_name=complainant_name or "Complainant",
        complainant_address=complainant_address or "Address not provided",
        incident_description=incident_description or "As described verbally",
        incident_date=incident_date or "Date not specified",
        accused_name=accused_name or "Unknown / To be determined",
        additional_details=additional_details or "None",
    )
    response = gemini.generate_content(prompt)
    return response.text.strip()


# ─── Q&A with Case Document ───────────────────────────────────────────────────

def _extract_law_context(question: str, law_db) -> str:
    if law_db is None:
        return "No legal knowledge base loaded."
    try:
        retriever = law_db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)
        return "\n\n".join(
            d.page_content.strip() for d in docs
            if (getattr(d, "page_content", "") or "").strip()
        ) or "No relevant legal provisions found."
    except Exception:
        logger.exception("Law retriever failed")
        return "Legal knowledge base retrieval failed."


def _build_case_context(case_data: dict) -> str:
    """Serialize case JSON to a clean, readable context string for the LLM."""
    if case_data.get("_raw"):
        return case_data.get("_raw_gemini", case_data.get("legal_summary", ""))

    lines = ["CASE DOCUMENT ANALYSIS:", ""]

    doc_type = case_data.get("document_type", "Legal Document")
    case_num = case_data.get("case_number")
    lines.append(f"Document Type: {doc_type}" + (f" (Case No: {case_num})" if case_num else ""))

    ps = case_data.get("police_station") or case_data.get("court")
    if ps:
        lines.append(f"Authority: {ps}")

    filing_date = case_data.get("date_of_filing")
    if filing_date:
        lines.append(f"Date of Filing: {filing_date}")

    complainant = case_data.get("complainant", {})
    if complainant and complainant.get("name"):
        lines.append(f"Complainant: {complainant['name']}" +
                     (f", {complainant['address']}" if complainant.get("address") else ""))

    victim = case_data.get("victim", {})
    if victim and victim.get("name"):
        lines.append(f"Victim: {victim['name']}")

    for acc in case_data.get("accused", []) or []:
        lines.append(f"Accused: {acc.get('name', 'Unknown')}" +
                     (f" ({acc.get('description', '')})" if acc.get("description") else ""))

    for p in case_data.get("parties", []) or []:
        lines.append(f"{p.get('role', 'Party')}: {p.get('name', '')}")

    incident = case_data.get("incident", {})
    if incident:
        if incident.get("date"):
            lines.append(f"Incident Date: {incident['date']}")
        if incident.get("time"):
            lines.append(f"Incident Time: {incident['time']}")
        if incident.get("location"):
            lines.append(f"Incident Location: {incident['location']}")
        if incident.get("description"):
            lines.append(f"Incident Description: {incident['description']}")

    sections = case_data.get("applicable_sections", [])
    if sections:
        lines.append("\nCharges / Applicable Sections:")
        for s in sections:
            lines.append(f"  - {s.get('act', '')} {s.get('section', '')}: {s.get('description', '')}")

    evidence = case_data.get("evidence_mentioned", [])
    if evidence:
        lines.append("\nEvidence on Record:")
        for e in evidence:
            lines.append(f"  - {e}")

    key_facts = case_data.get("key_facts", [])
    if key_facts:
        lines.append("\nKey Facts:")
        for f in key_facts:
            lines.append(f"  - {f}")

    strength = case_data.get("case_strength", {})
    if strength:
        lines.append(f"\nCase Strength: {strength.get('rating', 'Unknown')}")

    relief = case_data.get("relief_sought")
    if relief:
        lines.append(f"Relief Sought: {relief}")

    insights = case_data.get("legal_insights", {})
    if insights and insights.get("primary_offence"):
        lines.append(f"\nPrimary Offence: {insights['primary_offence']}")

    return "\n".join(lines)


class CaseQAChain:
    """Q&A chain for when a case document is loaded."""

    def __init__(self, case_data: dict, law_db):
        self.case_data = case_data
        self.law_db = law_db
        self._case_context = _build_case_context(case_data)

    def invoke(self, inputs: dict) -> dict:
        question = (inputs.get("question") or "").strip()
        if not question:
            return {"answer": "Please ask a legal question about the uploaded document."}

        law_context = _extract_law_context(question, self.law_db)
        prompt = CASE_QA_PROMPT_TEMPLATE.format(
            case_context=self._case_context,
            law_context=law_context,
            question=question,
        )
        try:
            result = _get_llm().invoke(prompt)
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception:
            logger.exception("CaseQAChain LLM invocation failed")
            raise
        return {"answer": answer}


def build_combined_chain(case_data: dict, law_db=None):
    """Build Q&A chain for a loaded case document."""
    if law_db is None:
        embeddings = _get_embeddings()
        index_dir = (Path(__file__).resolve().parent / ".faiss_index").resolve()
        if index_dir.exists():
            law_db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            logger.info("Law FAISS loaded from disk in build_combined_chain fallback")
        else:
            logger.warning("Law FAISS index not found. Legal provisions will be unavailable.")

    return CaseQAChain(case_data=case_data, law_db=law_db)


# ─── Conversation Persistence Helpers ────────────────────────────────────────

def new_conversation_id() -> str:
    return str(uuid.uuid4())


def make_message(role: str, content: str) -> dict:
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }


def history_from_messages(messages: list[dict]) -> list[tuple[str, str]]:
    """Convert message list to LangChain (user, assistant) tuple pairs."""
    pairs = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            pairs.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs


def generate_conversation_title(first_question: str) -> str:
    q = first_question.strip()
    if len(q) <= 50:
        return q
    return q[:47].rsplit(" ", 1)[0] + "..."


# ─── CLI ─────────────────────────────────────────────────────────────────────

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