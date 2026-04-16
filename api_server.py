import os
import threading
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot_backend import build_legal_chain, detect_output_language, translate_answer

app = FastAPI(title="NomoSys API", version="1.0.0")


def _parse_cors_origins() -> tuple[list[str], bool]:
    origins_env = os.getenv("CORS_ALLOW_ORIGINS")
    if origins_env is None:
        # Safe-by-default for local development; override in production.
        return ["http://localhost:3000", "http://localhost:5173"], True

    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    if not origins:
        return [], True

    if "*" in origins:
        # With wildcard origins, credentials must be disabled.
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
    # Lazy-load on first request so the service can start quickly on hosts like Render.
    app.state.qa_chain = None


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


class ChatResponse(BaseModel):
    answer: str
    target_lang: str


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_ready": getattr(app.state, "qa_chain", None) is not None}


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
        result = qa_chain.invoke({"question": req.question, "chat_history": req.history})
        answer = result.get("answer", "")

        target_lang = "en"
        if req.translate:
            target_lang = detect_output_language(req.question)
            if target_lang != "en":
                answer = translate_answer(answer, target_lang)

        return ChatResponse(answer=answer, target_lang=target_lang)
    except HTTPException:
        raise
    except Exception:
        # Avoid leaking internals.
        raise HTTPException(status_code=500, detail="Internal server error")
