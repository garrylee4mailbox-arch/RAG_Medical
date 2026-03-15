from __future__ import annotations

from fastapi import FastAPI, HTTPException
from .settings import settings
from .schemas import HealthResponse, IndexRequest, IndexResponse, AnswerRequest, AnswerResponse
from .rag_backend import RagBackend

app = FastAPI(title=settings.service, version=settings.version)
backend = RagBackend()

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        service=settings.service,
        version=settings.version,
        rag_ready=backend.rag_ready,
        index_info=backend.index_info if backend.rag_ready else {"data_sources": [], "built_at": None, "count": None},
    )

@app.post("/index", response_model=IndexResponse)
def index(req: IndexRequest):
    data = backend.build_index(req.data_sources, req.cache_dir, req.force_rebuild)
    return IndexResponse(**data)

@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    if not backend.rag_ready:
        raise HTTPException(status_code=409, detail="RAG_NOT_READY")
    try:
        data = backend.answer(req.question, req.top_n, req.score_threshold)
        return AnswerResponse(**data)
    except RuntimeError:
        # must match contract: HTTP 409 with {"error":"RAG_NOT_READY"}
        raise HTTPException(status_code=409, detail="RAG_NOT_READY")

# Custom exception handler to return {"error":"RAG_NOT_READY"} rather than {"detail":...}
from fastapi.responses import JSONResponse
from fastapi.requests import Request

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    if exc.status_code == 409 and str(exc.detail) == "RAG_NOT_READY":
        return JSONResponse(status_code=409, content={"error": "RAG_NOT_READY"})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
