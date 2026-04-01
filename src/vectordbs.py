#!/usr/bin/env python3
"""
BiomedBERT Embeddings → FAISS Vector Database
Improved pathology-aware chunking strategy
"""

import json
import pickle
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer


class PathologyVectorDB:

    def __init__(
        self,
        output_dir="output/biomedbert_vector_db",
        embedding_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    ):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading embedding model...")
        self.model = SentenceTransformer(embedding_model)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Embedding dimension: {self.embedding_dim}")

        self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.chunks = []

        self.stats = {
            "files_processed": 0,
            "total_chunks": 0,
        }

    # ----------------------------------------------------
    # TEXT NORMALIZATION
    # ----------------------------------------------------

    def normalize_text(self, text):

        text = text.replace("\r", "")
        text = text.replace("\n\n", "\n")

        return text.strip()

    # ----------------------------------------------------
    # SECTION SPLITTING
    # ----------------------------------------------------

    def split_by_sections(self, text):

        pattern = r"\n(?=[A-Z ]{4,}:?)"

        sections = re.split(pattern, text)

        return [s.strip() for s in sections if s.strip()]

    # ----------------------------------------------------
    # CHUNK CREATION
    # ----------------------------------------------------

    def create_chunks(self, text, filename,
                      chunk_size=1200,
                      overlap=200):

        text = self.normalize_text(text)

        sections = self.split_by_sections(text)

        chunks = []

        chunk_id = 0

        for section in sections:

            start = 0
            length = len(section)

            while start < length:

                end = start + chunk_size

                chunk_text = section[start:end]

                if len(chunk_text.strip()) > 50:

                    chunk = {
                        "chunk_id": chunk_id,
                        "filename": filename,
                        "text": chunk_text.strip(),
                    }

                    chunks.append(chunk)

                    chunk_id += 1

                start += chunk_size - overlap

        return chunks

    # ----------------------------------------------------
    # PROCESS SINGLE FILE
    # ----------------------------------------------------

    def process_file(self, txt_file):

        filename = txt_file.name

        text = txt_file.read_text(encoding="utf-8", errors="ignore")

        chunks = self.create_chunks(text, filename)

        if not chunks:
            return [], []

        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings, chunks

    # ----------------------------------------------------
    # ADD TO FAISS
    # ----------------------------------------------------

    def add_to_faiss(self, embeddings, chunks):

        embeddings = np.array(embeddings).astype("float32")

        start_idx = self.index.ntotal

        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):

            chunk["faiss_index"] = start_idx + i

            self.chunks.append(chunk)

    # ----------------------------------------------------
    # PROCESS DIRECTORY
    # ----------------------------------------------------

    def process_directory(self, text_dir):

        text_dir = Path(text_dir)

        files = list(text_dir.glob("*.txt"))

        if not files:
            print("No text files found.")
            return

        print(f"Found {len(files)} pathology reports")

        all_embeddings = []
        all_chunks = []

        for file in tqdm(files):

            embeddings, chunks = self.process_file(file)

            if len(chunks) == 0:
                continue

            all_embeddings.extend(embeddings)
            all_chunks.extend(chunks)

            self.stats["files_processed"] += 1

        if not all_chunks:
            print("No chunks created")
            return

        self.stats["total_chunks"] = len(all_chunks)

        embeddings_matrix = np.vstack(all_embeddings)

        print("Adding vectors to FAISS...")

        self.add_to_faiss(embeddings_matrix, all_chunks)

        self.save()

        self.print_summary()

    # ----------------------------------------------------
    # SAVE DATABASE
    # ----------------------------------------------------

    def save(self):

        index_file = self.output_dir / "faiss.index"

        faiss.write_index(self.index, str(index_file))

        metadata_file = self.output_dir / "metadata.pkl"

        with open(metadata_file, "wb") as f:

            pickle.dump(
                {
                    "chunks": self.chunks,
                    "embedding_dim": self.embedding_dim,
                    "model": "BiomedBERT",
                },
                f,
            )

        stats_file = self.output_dir / "stats.json"

        with open(stats_file, "w") as f:

            json.dump(
                {
                    **self.stats,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    # ----------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------

    def print_summary(self):

        print("\n=================================")
        print("VECTOR DATABASE CREATED")
        print("=================================")

        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total chunks: {self.stats['total_chunks']}")
        print(f"Vectors stored: {self.index.ntotal}")

        print("\nSaved to:")
        print(self.output_dir)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():

    TEXT_DIR = "output/text"
    VECTOR_DB = "output/biomedbert_vector_db"

    print("Pathology Vector DB Builder")

    pipeline = PathologyVectorDB(
        output_dir=VECTOR_DB
    )

    pipeline.process_directory(TEXT_DIR)


if __name__ == "__main__":
    main()