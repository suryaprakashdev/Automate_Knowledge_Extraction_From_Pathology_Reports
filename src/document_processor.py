#!/usr/bin/env python3
"""
Dynamic RAG Database Updater
Processes new PDFs and updates the vector database in real-time
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import pickle
from datetime import datetime

# PDF processing
from pdf2image import convert_from_path
from PIL import Image

# OCR (Tesseract – CPU, HF-safe)
import pytesseract

# Embeddings
from sentence_transformers import SentenceTransformer

# FAISS (CPU)
import faiss


class DynamicRAGUpdater:
    """
    Handles dynamic updates to RAG database:
    1. Upload PDF
    2. OCR extraction (Tesseract – CPU)
    3. Generate embeddings (BiomedBERT)
    4. Update FAISS index
    5. Update metadata
    """

    def __init__(
        self,
        vector_db_path: str,
        embedding_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        upload_dir: str = "uploaded_reports"
    ):
        self.vector_db_path = Path(vector_db_path)
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

        self.ocr_dir = self.upload_dir / "ocr_text"
        self.embeddings_dir = self.upload_dir / "embeddings"
        self.ocr_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)

        print("✅ Using Tesseract OCR (CPU)")

        # BiomedBERT embeddings (CPU)
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
            self.chunk_id_to_idx = data["chunk_id_to_idx"]

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

    # ---------------- OCR (Tesseract) ---------------- #

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        images = convert_from_path(pdf_path, dpi=300)

        full_text = []

        for page_num, image in enumerate(images, 1):
            page_text = pytesseract.image_to_string(
                image,
                lang="eng",
                config="--oem 3 --psm 6"
            )

            full_text.append(
                f"\n{'='*50}\nPAGE {page_num}\n{'='*50}\n{page_text}"
            )

        return "\n".join(full_text)

    # ---------------- Chunking ---------------- #

    def chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        sentences = text.split(". ")
        chunks = []
        current = []
        length = 0

        for s in sentences:
            s = s.strip()
            if not s:
                continue

            s = s + ". "
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

    # ---------------- Embeddings ---------------- #

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        return self.embedding_model.encode(
            chunks,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    # ---------------- FAISS Update ---------------- #

    def add_to_database(
        self,
        embeddings: np.ndarray,
        chunks: List[str],
        filename: str
    ) -> int:
        start_idx = self.faiss_index.ntotal
        self.faiss_index.add(embeddings.astype("float32"))

        for i, text in enumerate(chunks):
            meta = {
                "chunk_id": start_idx + i,
                "text": text,
                "filename": filename,
                "upload_date": datetime.now().isoformat(),
                "source": "user_upload"
            }
            self.chunks.append(meta)
            self.chunk_id_to_idx[f"{filename}_{i}"] = start_idx + i

        return len(embeddings)

    # ---------------- Full Pipeline ---------------- #

    def process_and_add_pdf(self, pdf_path: str) -> Dict:
        start = datetime.now()
        filename = Path(pdf_path).stem

        text = self.extract_text_from_pdf(pdf_path)
        (self.ocr_dir / f"{filename}.txt").write_text(text, encoding="utf-8")

        chunks = self.chunk_text(text)
        embeddings = self.generate_embeddings(chunks)

        np.save(self.embeddings_dir / f"{filename}_embeddings.npy", embeddings)

        vectors_added = self.add_to_database(embeddings, chunks, filename)
        self.save_database()

        return {
            "filename": filename,
            "text_length": len(text),
            "num_chunks": len(chunks),
            "vectors_added": vectors_added,
            "total_vectors": self.faiss_index.ntotal,
            "processing_time_seconds": (datetime.now() - start).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }


def main():
    vector_db_path = "output/biomedbert_vector_db"

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
