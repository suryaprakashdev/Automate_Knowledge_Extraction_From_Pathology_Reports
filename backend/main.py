"""FastAPI app for the Pathology RAG system.

Serves a streaming chat API over the existing RAG pipeline and (in the
container) the built React SPA from the same origin.
"""

import os
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .schemas import (
    ChatRequest, ReportsResponse, HealthResponse, UploadStartResponse,
)
from .rag_service import service, UPLOAD_DIR

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the (expensive) pipeline once at startup.
    service.load()
    yield


app = FastAPI(title="Pathology RAG API", lifespan=lifespan)

# Permissive CORS only matters for local dev (Vite on :5173 → API on :8000).
# In the container the SPA is same-origin so this is a no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _sse(threaded_stream):
    """Adapt a ThreadedEventStream ({"event","data":dict}) to the dict shape
    EventSourceResponse expects ({"event", "data": <json str>})."""
    async for item in threaded_stream:
        yield {"event": item["event"], "data": json.dumps(item["data"])}


# ── API ─────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health():
    if service.ready:
        return HealthResponse(status="ok", num_documents=len(service.get_reports()))
    return HealthResponse(status="degraded", num_documents=0, detail=service.load_error)


@app.get("/api/reports", response_model=ReportsResponse)
async def reports():
    return ReportsResponse(reports=service.get_reports())


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not service.ready:
        raise HTTPException(503, detail=f"Pipeline not ready: {service.load_error}")

    history = [m.model_dump() for m in req.history] if req.history else None
    producer = service.make_chat_producer(
        question=req.question,
        report_name=req.report_name,
        report_names=req.report_names,
        top_k=req.top_k,
        history=history,
    )
    from .rag_service import ThreadedEventStream
    stream = ThreadedEventStream()
    stream.run(producer)
    return EventSourceResponse(_sse(stream))


@app.post("/api/upload", response_model=UploadStartResponse)
async def upload(file: UploadFile = File(...)):
    if not service.ready:
        raise HTTPException(503, detail=f"Pipeline not ready: {service.load_error}")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are supported.")

    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    # Strip any path components from the client-supplied name.
    safe_name = Path(file.filename).name
    dest = upload_dir / safe_name
    dest.write_bytes(await file.read())

    job_id = service.start_upload(str(dest))
    return UploadStartResponse(job_id=job_id, filename=safe_name)


@app.get("/api/upload/{job_id}/events")
async def upload_events(job_id: str):
    stream = service.get_job(job_id)
    if stream is None:
        raise HTTPException(404, detail="Unknown or expired upload job.")

    async def gen():
        async for evt in _sse(stream):
            yield evt
        service.pop_job(job_id)

    return EventSourceResponse(gen())


@app.get("/api/pdf/{name}")
async def get_pdf(name: str):
    """Serve a raw uploaded PDF for the viewer. `name` may be the report stem
    or the full file name; path components are stripped to prevent traversal."""
    stem = Path(name).name
    upload_dir = Path(UPLOAD_DIR)
    candidate = upload_dir / (stem if stem.lower().endswith(".pdf") else f"{stem}.pdf")
    if not candidate.exists():
        raise HTTPException(404, detail="PDF not found.")
    return FileResponse(str(candidate), media_type="application/pdf")


# ── Static SPA (only when a build exists; in dev Vite serves the frontend) ──
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="spa")
