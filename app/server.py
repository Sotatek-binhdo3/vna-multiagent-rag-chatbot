"""Backend FastAPI cho POC.

- POST /api/chat        : entrypoint chat, di qua orchestrator agent
- GET  /api/page/{n}    : anh trang PDF, dung cho preview citation tren UI
- GET  /api/health      : kiem tra cau hinh + trang thai collection Qdrant
- GET  /                : serve giao dien chat

Chay:  python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import traceback
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, session as session_store
from app.agents.orchestrator import Orchestrator

app = FastAPI(title="VNA Technical Chatbot - POC Multi-Agent RAG")

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ResetRequest(BaseModel):
    session_id: str


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    session_id = req.session_id or str(uuid.uuid4())
    sess = session_store.get(session_id)
    try:
        result = get_orchestrator().handle(req.message.strip(), sess)
    except Exception as exc:  # tra loi ro rang thay vi 500 trang trong khi demo
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    result["session_id"] = session_id
    return JSONResponse(result)


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict[str, str]:
    session_store.reset(req.session_id)
    return {"status": "ok"}


@app.get("/api/page/{page_index}")
def page_image(page_index: int) -> FileResponse:
    path = config.PAGES_DIR / f"page_{page_index:03d}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Khong co anh cho trang PDF {page_index}")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/health")
def health() -> dict[str, object]:
    status: dict[str, object] = {
        "source_file": config.SOURCE_NAME,
        "provider": config.PROVIDER,
        "chat_model": config.CHAT_MODEL,
        "fast_model": config.FAST_MODEL,
        "embed_model": config.GEMINI_EMBED_MODEL,
        "rerank_model": config.COHERE_RERANK_MODEL,
        "keys": {
            "gemini_embedding": bool(config.GEMINI_API_KEY) and not config.GEMINI_API_KEY.startswith("your_"),
            "openrouter": bool(config.OPENROUTER_API_KEY) and not config.OPENROUTER_API_KEY.startswith("your_"),
            "qdrant": bool(config.QDRANT_API_KEY) and not config.QDRANT_API_KEY.startswith("your_"),
            "cohere": bool(config.COHERE_API_KEY) and not config.COHERE_API_KEY.startswith("your_"),
        },
        "page_images": len(list(config.PAGES_DIR.glob("*.jpg"))) if config.PAGES_DIR.exists() else 0,
        "parsed": config.PARSED_JSONL.exists(),
    }
    try:
        from app import vector_store

        status["qdrant_mode"] = vector_store.describe()
        client = vector_store.get_client(timeout=10)
        if client.collection_exists(config.QDRANT_COLLECTION):
            status["qdrant_points"] = client.get_collection(config.QDRANT_COLLECTION).points_count
        else:
            status["qdrant_points"] = None
            status["qdrant_note"] = f"Collection '{config.QDRANT_COLLECTION}' chua ton tai"
    except Exception as exc:
        status["qdrant_points"] = None
        status["qdrant_note"] = f"Khong ket noi duoc Qdrant: {type(exc).__name__}"
    return status


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")
