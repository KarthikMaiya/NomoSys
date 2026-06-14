import io
import logging
import os
import base64
import threading
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot_backend import (
    build_legal_chain,
    detect_output_language,
    translate_answer,
    analyze_case_document,
    build_combined_chain,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="NomoSys API", version="2.0.0")


def _parse_cors_origins() -> tuple[list[str], bool]:
    origins_env = os.getenv("CORS_ALLOW_ORIGINS")
    if origins_env is None:
        return ["http://localhost:3000", "http://localhost:5173"], True
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


_qa_chain_lock = threading.Lock()


@app.on_event("startup")
def _startup() -> None:
    app.state.qa_chain = None
    app.state.case_db = None
    app.state.case_summary = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="List of (user_question, assistant_answer) tuples.",
    )
    translate: bool = Field(
        default=True,
        description="If true, detects 'in Hindi/Telugu/...' in the question and translates the answer.",
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Name of uploaded case file (for type detection)",
    )
    file_content: Optional[str] = Field(
        default=None,
        description="Base64-encoded file content for inline upload",
    )


class ChatResponse(BaseModel):
    answer: str
    target_lang: str
    case_summary: Optional[str] = Field(
        default=None,
        description="Case summary if a document was analyzed",
    )


class UploadResponse(BaseModel):
    summary: str
    status: str = "ok"
    chunks: int = Field(description="Number of document chunks created")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "model_ready": getattr(app.state, "qa_chain", None) is not None,
        "case_loaded": getattr(app.state, "case_db", None) is not None,
    }


@app.post("/upload", response_model=UploadResponse)
def upload_case(file: UploadFile = File(...)):
    """Upload a case document (PDF/TXT) for analysis. Returns a structured case summary."""
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    logger.info(
        "Received upload request: file_name=%s content_type=%s bytes=%d",
        file.filename,
        file.content_type,
        len(content),
    )
    try:
        file_obj = io.BytesIO(content)
        file_obj.name = file.filename or "document.pdf"
        case_db, summary = analyze_case_document(file_obj)
        with _qa_chain_lock:
            app.state.case_db = case_db
            app.state.case_summary = summary
        return UploadResponse(
            summary=summary,
            chunks=case_db.index.ntotal,
        )
    except ValueError as e:
        logger.warning("Document upload rejected: file_name=%s error=%s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Document upload failed unexpectedly: file_name=%s", file.filename)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")


@app.delete("/case")
def clear_case():
    """Clear the uploaded case document and its FAISS index."""
    with _qa_chain_lock:
        app.state.case_db = None
        app.state.case_summary = None
    return {"status": "cleared"}


@app.get("/case")
def get_case_status():
    """Check if a case document is currently loaded."""
    has_case = getattr(app.state, "case_db", None) is not None
    return {
        "has_case": has_case,
        "summary": getattr(app.state, "case_summary", None) if has_case else None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    qa_chain = getattr(app.state, "qa_chain", None)
    if qa_chain is None:
        with _qa_chain_lock:
            qa_chain = getattr(app.state, "qa_chain", None)
            if qa_chain is None:
                try:
                    qa_chain = build_legal_chain()
                    app.state.qa_chain = qa_chain
                except Exception:
                    raise HTTPException(status_code=503, detail="Model failed to initialize")

    try:
        # Handle inline file upload
        if req.file_content:
            try:
                file_bytes = base64.b64decode(req.file_content)
                logger.info(
                    "Processing inline uploaded file: file_name=%s bytes=%d",
                    req.file_name,
                    len(file_bytes),
                )
                file_obj = io.BytesIO(file_bytes)
                file_obj.name = req.file_name or "document.pdf"
                case_db, case_summary = analyze_case_document(file_obj)
                with _qa_chain_lock:
                    app.state.case_db = case_db
                    app.state.case_summary = case_summary
            except ValueError as e:
                logger.warning("Inline file rejected: file_name=%s error=%s", req.file_name, e)
                raise HTTPException(status_code=400, detail=str(e))
            except Exception:
                logger.exception("Inline file processing failed: file_name=%s", req.file_name)
                raise HTTPException(status_code=400, detail="Failed to process inline file")

        # Use combined chain if case document is loaded
        case_db = getattr(app.state, "case_db", None)
        if case_db is not None:
            try:
                chain = build_combined_chain(case_db)
                result = chain.invoke({"question": req.question, "chat_history": req.history})
            except Exception:
                raise HTTPException(status_code=500, detail="Combined chain error")
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
        raise HTTPException(status_code=500, detail="Internal server error")
