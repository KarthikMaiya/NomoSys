"""NomoSys API Server — v5.0 Demo-Day Edition.

Changes vs v4:
- /complaint endpoint for complaint letter generation
- /case returns timeline and legal_insights
- Chat history is passed through correctly per-request
- All state management unchanged (per-process singletons)
"""
import io
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot_backend import (
    build_legal_chain,
    detect_output_language,
    translate_answer,
    analyze_case_document,
    build_combined_chain,
    generate_legal_complaint,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="NomoSys API", version="5.0.0")


# ─── CORS ─────────────────────────────────────────────────────────────────────

def _parse_cors_origins() -> tuple[list[str], bool]:
    origins_env = os.getenv("CORS_ALLOW_ORIGINS")
    if origins_env is None:
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:5176",
        ], True
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    if not origins:
        return [], True
    if "*" in origins:
        return ["*"], False
    return origins, True


_cors_origins, _cors_allow_credentials = _parse_cors_origins()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ─── State Lock ───────────────────────────────────────────────────────────────

_qa_chain_lock = threading.Lock()


# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup() -> None:
    app.state.qa_chain = None
    app.state.law_db = None
    app.state.case_data = None
    app.state.case_summary = None
    app.state.combined_chain = None

    def _warmup():
        try:
            logger.info("Pre-warming legal chain + embeddings + LLM...")
            chain, law_db = build_legal_chain()
            with _qa_chain_lock:
                app.state.qa_chain = chain
                app.state.law_db = law_db
            logger.info("Pre-warm complete. Ready.")
        except Exception:
            logger.exception("Pre-warm failed (will lazy-init on first request)")

    threading.Thread(target=_warmup, daemon=True).start()


# ─── Models ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="List of (user_question, assistant_answer) tuples.",
    )
    translate: bool = Field(default=True)


class ChatResponse(BaseModel):
    answer: str
    target_lang: str
    case_summary: Optional[str] = Field(default=None)


class UploadResponse(BaseModel):
    summary: str
    status: str = "ok"
    document_type: Optional[str] = Field(default=None)
    case_number: Optional[str] = Field(default=None)
    case_strength: Optional[str] = Field(default=None)
    timeline: Optional[list] = Field(default=None)
    legal_insights: Optional[dict] = Field(default=None)


class CaseStatusResponse(BaseModel):
    has_case: bool
    summary: Optional[str] = None
    document_type: Optional[str] = None
    case_number: Optional[str] = None
    case_strength: Optional[str] = None
    timeline: Optional[list] = None
    legal_insights: Optional[dict] = None


class ComplaintRequest(BaseModel):
    complaint_type: str = Field(..., description="e.g. Cyberbullying, Fraud, Harassment")
    complainant_name: str
    complainant_address: str
    incident_description: str
    incident_date: str = ""
    accused_name: str = ""
    additional_details: str = ""


class ComplaintResponse(BaseModel):
    letter: str
    status: str = "ok"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "model_ready": getattr(app.state, "qa_chain", None) is not None,
        "case_loaded": getattr(app.state, "case_data", None) is not None,
        "version": "5.0.0",
    }


@app.post("/upload", response_model=UploadResponse)
def upload_case(file: UploadFile = File(...)):
    """Upload a case document (PDF/TXT). Returns structured summary from Gemini."""
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("Upload: file=%s content_type=%s bytes=%d",
                file.filename, file.content_type, len(content))

    try:
        file_obj = io.BytesIO(content)
        file_obj.name = file.filename or "document.pdf"

        case_data, summary = analyze_case_document(file_obj)

        law_db = getattr(app.state, "law_db", None)
        combined_chain = build_combined_chain(case_data, law_db=law_db)

        with _qa_chain_lock:
            app.state.case_data = case_data
            app.state.case_summary = summary
            app.state.combined_chain = combined_chain

        strength = None
        if isinstance(case_data.get("case_strength"), dict):
            strength = case_data["case_strength"].get("rating")

        from chatbot_backend import _extract_case_timeline
        timeline = _extract_case_timeline(case_data)

        return UploadResponse(
            summary=summary,
            document_type=case_data.get("document_type"),
            case_number=case_data.get("case_number"),
            case_strength=strength,
            timeline=timeline or None,
            legal_insights=case_data.get("legal_insights") or None,
        )
    except ValueError as e:
        logger.warning("Upload rejected: file=%s error=%s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Upload failed: file=%s", file.filename)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")


@app.delete("/case")
def clear_case():
    with _qa_chain_lock:
        app.state.case_data = None
        app.state.case_summary = None
        app.state.combined_chain = None
    return {"status": "cleared"}


@app.get("/case", response_model=CaseStatusResponse)
def get_case_status():
    case_data = getattr(app.state, "case_data", None)
    if case_data is None:
        return CaseStatusResponse(has_case=False)

    strength = None
    if isinstance(case_data.get("case_strength"), dict):
        strength = case_data["case_strength"].get("rating")

    from chatbot_backend import _extract_case_timeline
    timeline = _extract_case_timeline(case_data)

    return CaseStatusResponse(
        has_case=True,
        summary=getattr(app.state, "case_summary", None),
        document_type=case_data.get("document_type"),
        case_number=case_data.get("case_number"),
        case_strength=strength,
        timeline=timeline or None,
        legal_insights=case_data.get("legal_insights") or None,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    qa_chain = getattr(app.state, "qa_chain", None)
    if qa_chain is None:
        with _qa_chain_lock:
            qa_chain = getattr(app.state, "qa_chain", None)
            if qa_chain is None:
                try:
                    chain, law_db = build_legal_chain()
                    app.state.qa_chain = chain
                    app.state.law_db = law_db
                    qa_chain = chain
                except Exception:
                    raise HTTPException(status_code=503, detail="Model failed to initialize")
    else:
        logger.info("CHAIN REUSED")

    try:
        combined_chain = getattr(app.state, "combined_chain", None)
        if combined_chain is not None:
            result = combined_chain.invoke({"question": req.question, "chat_history": req.history})
        else:
            result = qa_chain.invoke({"question": req.question, "chat_history": req.history})

        answer = result.get("answer", "")

        target_lang = "en"
        if req.translate:
            target_lang = detect_output_language(req.question)
            if target_lang != "en":
                answer = translate_answer(answer, target_lang)

        return ChatResponse(
            answer=answer,
            target_lang=target_lang,
            case_summary=getattr(app.state, "case_summary", None),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/complaint", response_model=ComplaintResponse)
def generate_complaint(req: ComplaintRequest) -> ComplaintResponse:
    """Generate a formal Indian legal complaint letter."""
    try:
        letter = generate_legal_complaint(
            complaint_type=req.complaint_type,
            complainant_name=req.complainant_name,
            complainant_address=req.complainant_address,
            incident_description=req.incident_description,
            incident_date=req.incident_date,
            accused_name=req.accused_name,
            additional_details=req.additional_details,
        )
        return ComplaintResponse(letter=letter)
    except Exception as e:
        logger.exception("Complaint generation failed")
        raise HTTPException(status_code=500, detail=f"Complaint generation failed: {e}")