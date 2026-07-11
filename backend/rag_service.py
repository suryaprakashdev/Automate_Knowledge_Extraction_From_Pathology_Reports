"""Service layer around the existing RAG pipeline.

Wraps `CompleteRAGPipeline` (retrieval + streaming generation) and
`DynamicRAGUpdater` (OCR upload + index update), and provides a small
thread→async bridge so the blocking ML work streams cleanly over SSE
without blocking the event loop.
"""

import os
import sys
import uuid
import queue
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable

# Reuse the existing, working RAG code in src/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

DB_PATH = os.getenv("DB_PATH", "output/biomedbert_vector_db")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_reports")


# ────────────────────────────────────────────────────────────────────────────
# Thread → async SSE bridge
# ────────────────────────────────────────────────────────────────────────────
class ThreadedEventStream:
    """Runs a blocking `producer(emit)` in a daemon thread and exposes an async
    iterator of the events it emits. The producer calls `emit({"event","data"})`
    for each event; iteration ends when the producer returns (or raises, which
    is surfaced as a final error event)."""

    _DONE = object()

    def __init__(self):
        self._q: "queue.Queue" = queue.Queue()

    def run(self, producer: Callable[[Callable[[dict], None]], None]):
        def worker():
            try:
                producer(self._q.put)
            except Exception as exc:  # noqa: BLE001 - surface to the client
                self._q.put({"event": "error", "data": {"message": str(exc)}})
            finally:
                self._q.put(self._DONE)

        threading.Thread(target=worker, daemon=True).start()

    async def __aiter__(self):
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if item is self._DONE:
                break
            yield item


class RagService:
    def __init__(self):
        self.pipeline = None
        self._updater = None
        self._ocr_engine = None
        self.load_error: Optional[str] = None
        self._jobs: Dict[str, ThreadedEventStream] = {}
        self._job_files: Dict[str, str] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────
    def load(self):
        """Load the pipeline once at startup. Failures (missing API key / DB)
        are captured so the app can still serve the frontend and report a
        degraded health status instead of crashing."""
        try:
            from retriever import CompleteRAGPipeline
            self.pipeline = CompleteRAGPipeline(
                faiss_db_path=DB_PATH,
                embedding_model=EMBEDDING_MODEL,
            )
            self.load_error = None
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.pipeline = None
            self.load_error = str(exc)
            # Print the full traceback so the real failing import shows up in
            # container logs instead of just the top-level message.
            print("=== PIPELINE LOAD FAILED ===", flush=True)
            traceback.print_exc()

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def _get_updater(self):
        """Lazily build the OCR-based updater on first upload, reusing the
        pipeline's already-loaded embedder so only one embedding model lives
        in memory (same pattern the Streamlit app used)."""
        if self._updater is None:
            from paddleocr import PaddleOCR
            from document_processor import DynamicRAGUpdater
            if self._ocr_engine is None:
                # PaddleOCR 3.x API (see document_processor for rationale).
                self._ocr_engine = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            self._updater = DynamicRAGUpdater(
                vector_db_path=DB_PATH,
                upload_dir=UPLOAD_DIR,
                ocr_engine=self._ocr_engine,
                embedder=self.pipeline.query_processor.model,
            )
        return self._updater

    # ── reads ──────────────────────────────────────────────────────────────
    def get_reports(self) -> List[str]:
        if not self.ready:
            return []
        return self.pipeline.get_available_reports()

    # ── retrieval + streaming generation ────────────────────────────────────
    @staticmethod
    def _sources_payload(top_chunks: List[Dict]) -> List[dict]:
        out = []
        for i, src in enumerate(top_chunks, 1):
            chunk = src["chunk"]
            out.append({
                "index": i,
                "filename": chunk.get("filename", "unknown"),
                "page": chunk.get("page"),
                "line_bboxes": chunk.get("line_bboxes", []),
                "text": chunk.get("text", ""),
                "score": float(src.get("ce_score", src.get("score", 0.0))),
            })
        return out

    def make_chat_producer(
        self,
        question: str,
        report_name: Optional[str],
        report_names: Optional[List[str]],
        top_k: int,
        history: Optional[List[dict]],
    ):
        """Returns a producer(emit) that: retrieves → emits `sources` →
        streams `token` deltas → emits `done`."""

        def producer(emit: Callable[[dict], None]):
            top_chunks = self.pipeline.retrieve(
                question,
                report_name=report_name,
                report_names=report_names,
                top_k=top_k,
            )
            emit({"event": "sources", "data": {"sources": self._sources_payload(top_chunks)}})

            if not top_chunks:
                emit({"event": "token", "data": {"text": "No relevant information found for this question."}})
                emit({"event": "done", "data": {"num_sources": 0}})
                return

            for delta in self.pipeline.llm.generate_stream(question, top_chunks, history):
                emit({"event": "token", "data": {"text": delta}})

            emit({"event": "done", "data": {"num_sources": len(top_chunks)}})

        return producer

    # ── upload as a background job with progress SSE ─────────────────────────
    def start_upload(self, pdf_path: str) -> str:
        job_id = uuid.uuid4().hex
        stream = ThreadedEventStream()

        def producer(emit: Callable[[dict], None]):
            def cb(stage: str, detail: dict):
                emit({"event": "progress", "data": {"stage": stage, **(detail or {})}})

            updater = self._get_updater()
            stats = updater.process_and_add_pdf(pdf_path, progress_callback=cb)
            self.pipeline.reload_index()
            emit({"event": "complete", "data": stats})

        stream.run(producer)
        self._jobs[job_id] = stream
        self._job_files[job_id] = pdf_path
        return job_id

    def get_job(self, job_id: str) -> Optional[ThreadedEventStream]:
        return self._jobs.get(job_id)

    def pop_job(self, job_id: str):
        self._jobs.pop(job_id, None)
        self._job_files.pop(job_id, None)


service = RagService()
