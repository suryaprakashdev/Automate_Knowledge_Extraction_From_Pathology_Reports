#!/usr/bin/env python3
"""
Dynamic RAG Database Updater
Processes new PDFs and updates the vector database in real-time

"""

# Disable PaddleOCR's connectivity check — models are already cached after
# first run; this check adds ~10-30s to every cold start for nothing.
import os
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import pickle
from datetime import datetime

# PDF processing
import fitz

# OCR (CPU optimized)
from paddleocr import PaddleOCR

# Embeddings
from sentence_transformers import SentenceTransformer

# FAISS (CPU)
import faiss


class DynamicRAGUpdater:
    """
    Handles dynamic updates to RAG database:
    1. Upload PDF
    2. OCR extraction (PaddleOCR CPU)
    3. Generate embeddings (BiomedBERT)
    4. Update FAISS index
    5. Update metadata
    """

    def __init__(
        self,
        vector_db_path: str,
        embedding_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        upload_dir: str = "uploaded_reports",
        ocr_engine=None,
        embedder=None,
        ocr_dpi: int = 300,
    ):
        self.vector_db_path = Path(vector_db_path)
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.ocr_dpi = ocr_dpi

        self.ocr_dir = self.upload_dir / "ocr_text"
        self.embeddings_dir = self.upload_dir / "embeddings"
        self.ocr_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)

        # Both models are expensive to load (PaddleOCR init + BiomedBERT weights).
        # Callers (e.g. the Streamlit app) should build these once via a cached
        # resource and pass them in so repeated uploads reuse warm models instead
        # of reloading from disk every time.
        if ocr_engine is not None:
            self.ocr = ocr_engine
        else:
            # PaddleOCR (CPU mode — mkldnn disabled: causes PIR attribute crash
            # with ConvertPirAttribute2RuntimeAttribute on some paddle builds)
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                cpu_threads=4,
                enable_mkldnn=False
            )

        if embedder is not None:
            self.embedding_model = embedder
        else:
            self.embedding_model = SentenceTransformer(
                embedding_model,
                device="cpu"
            )

        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.load_database()

    def load_database(self):
        index_file = self.vector_db_path / "faiss.index"
        metadata_file = self.vector_db_path / "metadata.pkl"

        self.faiss_index = faiss.read_index(str(index_file))

        with open(metadata_file, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.chunk_id_to_idx = data.get("chunk_id_to_idx", {})

    def save_database(self):
        faiss.write_index(self.faiss_index, str(self.vector_db_path / "faiss.index"))

        with open(self.vector_db_path / "metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "chunk_id_to_idx": self.chunk_id_to_idx,
                    "embedding_dim": self.embedding_dim,
                    "model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
                },
                f
            )

    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Runs OCR page by page, keeping each line's page number and bounding
        box (converted from pixmap pixel space back to PDF point space) so
        chunks can later be traced back to a highlightable location.
        Returns {"full_text": str, "lines": [{"page", "text", "bbox"}, ...]}.
        """
        doc = fitz.open(pdf_path)
        scale = self.ocr_dpi / 72.0  # pixmap pixels -> PDF points

        full_text_parts = []
        all_lines = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=self.ocr_dpi, alpha=False)
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            ocr_result = self.ocr.ocr(image_np)

            page_text = []
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    bbox_poly, (line_text, _conf) = line
                    xs = [pt[0] for pt in bbox_poly]
                    ys = [pt[1] for pt in bbox_poly]
                    bbox_pdf = [
                        min(xs) / scale, min(ys) / scale,
                        max(xs) / scale, max(ys) / scale,
                    ]
                    page_text.append(line_text)
                    all_lines.append({
                        "page": page_num + 1,
                        "text": line_text,
                        "bbox": bbox_pdf,
                    })

            full_text_parts.append(
                f"\n{'='*50}\nPAGE {page_num + 1}\n{'='*50}\n" +
                "\n".join(page_text)
            )

        return {
            "full_text": "\n".join(full_text_parts),
            "lines": all_lines,
        }

    def chunk_text(self, lines: List[Dict], chunk_size: int = 512) -> List[Dict]:
        """Groups OCR lines into ~chunk_size-character chunks, sentence-aware
        like before, but keeps each chunk's originating page and the bounding
        boxes of every line that fell inside it for PDF highlighting.
        Returns [{"text", "page", "line_bboxes"}, ...].
        """
        chunks = []
        current_text = []
        current_bboxes = []
        current_page = None
        length = 0

        def flush():
            if current_text:
                chunks.append({
                    "text": "".join(current_text),
                    "page": current_page,
                    "line_bboxes": list(current_bboxes),
                })

        for line in lines:
            s = line["text"].strip()
            if not s:
                continue
            s = s + ". "

            if length + len(s) > chunk_size and current_text:
                flush()
                current_text = [s]
                current_bboxes = [line["bbox"]]
                current_page = line["page"]
                length = len(s)
            else:
                if current_page is None:
                    current_page = line["page"]
                current_text.append(s)
                current_bboxes.append(line["bbox"])
                length += len(s)

        flush()
        return chunks

    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        return self.embedding_model.encode(
            [c["text"] for c in chunks],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,   # must match vectordbs.py (IndexFlatIP)
            show_progress_bar=True
        )

    def add_to_database(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict],
        filename: str
    ) -> int:
        start_idx = self.faiss_index.ntotal
        self.faiss_index.add(embeddings.astype("float32"))

        for i, chunk in enumerate(chunks):
            meta = {
                "chunk_id": start_idx + i,
                "text": chunk["text"],
                "filename": filename,
                "upload_date": datetime.now().isoformat(),
                "source": "user_upload",
                "page": chunk.get("page"),
                "line_bboxes": chunk.get("line_bboxes", []),
            }
            self.chunks.append(meta)
            self.chunk_id_to_idx[f"{filename}_{i}"] = start_idx + i

        return len(embeddings)

    def process_and_add_pdf(self, pdf_path: str) -> Dict:
        start = datetime.now()
        filename = Path(pdf_path).stem

        extracted = self.extract_text_from_pdf(pdf_path)
        full_text = extracted["full_text"]
        (self.ocr_dir / f"{filename}.txt").write_text(full_text, encoding="utf-8")

        chunks = self.chunk_text(extracted["lines"])
        embeddings = self.generate_embeddings(chunks)

        np.save(self.embeddings_dir / f"{filename}_embeddings.npy", embeddings)

        vectors_added = self.add_to_database(embeddings, chunks, filename)
        self.save_database()

        return {
            "filename": filename,
            "text_length": len(full_text),
            "num_chunks": len(chunks),
            "vectors_added": vectors_added,
            "total_vectors": self.faiss_index.ntotal,
            "processing_time_seconds": (datetime.now() - start).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }


def main():
    vector_db_path = "/usr/users/3d_dimension_est/selva_sur/RAG/output/biomedbert_vector_db"

    updater = DynamicRAGUpdater(
        vector_db_path=vector_db_path,
        embedding_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        upload_dir="uploaded_reports"
    )

    test_pdf = "path/to/new_report.pdf"

    if Path(test_pdf).exists():
        stats = updater.process_and_add_pdf(test_pdf)
        print(json.dumps(stats, indent=2))
    else:
        print("Test PDF not found. Update the path in main().")


if __name__ == "__main__":
    main()