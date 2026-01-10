#!/usr/bin/env python3
"""
Dynamic RAG Database Updater 
"""

import numpy as np
from pathlib import Path
from typing import List, Dict
import pickle
from datetime import datetime

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from sentence_transformers import SentenceTransformer
import faiss


class DynamicRAGUpdater:
    def __init__(
        self,
        vector_db_path: str,
        embedding_model: str,
    ):
        self.model_name = embedding_model

        self.vector_db_path = Path(vector_db_path)

        self.embedding_model = SentenceTransformer(
            embedding_model,
            device="cpu",
        )

        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.load_database()

    # --------------------------------------------------
    def load_database(self):
        self.faiss_index = faiss.read_index(str(self.vector_db_path / "faiss.index"))
        with open(self.vector_db_path / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.chunk_id_to_idx = data["chunk_id_to_idx"]

    def save_database(self):
        faiss.write_index(self.faiss_index, str(self.vector_db_path / "faiss.index"))
        with open(self.vector_db_path / "metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "chunk_id_to_idx": self.chunk_id_to_idx,
                    "embedding_dim": self.embedding_dim,
                    "model": self.model_name,

                },
                f,
            )

    # --------------------------------------------------
    # PDF → TEXT (NO POPPLER)
    # --------------------------------------------------
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        pages = []

        for i, page in enumerate(doc, 1):
            text = page.get_text().strip()

            # If digital text exists → use it
            if text:
                pages.append(f"\n{'='*40}\nPAGE {i}\n{'='*40}\n{text}")
                continue

            # OCR fallback (scanned PDF)
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)

            pages.append(f"\n{'='*40}\nPAGE {i}\n{'='*40}\n{ocr_text}")

        return "\n".join(pages)

    # --------------------------------------------------
    def chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        sentences = text.split(". ")
        chunks, current, length = [], [], 0

        for s in sentences:
            s = s.strip()
            if not s:
                continue

            s += ". "
            if length + len(s) > chunk_size and current:
                chunks.append("".join(current))
                current = [s]
                length = len(s)
            else:
                current.append(s)
                length += len(s)

        if current:
            chunks.append("".join(current))

        return chunks

    # --------------------------------------------------
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        return self.embedding_model.encode(
            chunks,
            batch_size=32,
            convert_to_numpy=True,
        )

    # --------------------------------------------------
    def add_to_database(
        self,
        embeddings: np.ndarray,
        chunks: List[str],
        filename: str,
    ) -> int:
        start_idx = self.faiss_index.ntotal
        self.faiss_index.add(embeddings.astype("float32"))

        for i, text in enumerate(chunks):
            self.chunks.append(
                {
                    "chunk_id": start_idx + i,
                    "text": text,
                    "filename": filename,
                    "upload_date": datetime.now().isoformat(),
                    "source": "user_upload",
                }
            )
            self.chunk_id_to_idx[f"{filename}_{i}"] = start_idx + i

        return len(embeddings)

    # --------------------------------------------------
    def process_and_add_pdf(self, pdf_path: str) -> Dict:
        start = datetime.now()
        filename = Path(pdf_path).stem

        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        embeddings = self.generate_embeddings(chunks)

        vectors_added = self.add_to_database(embeddings, chunks, filename)
        self.save_database()

        return {
            "filename": filename,
            "num_chunks": len(chunks),
            "vectors_added": vectors_added,
            "total_vectors": self.faiss_index.ntotal,
            "processing_time_seconds": (datetime.now() - start).total_seconds(),
        }
