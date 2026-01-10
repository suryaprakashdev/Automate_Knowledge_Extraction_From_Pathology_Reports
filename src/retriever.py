#!/usr/bin/env python3
"""
Complete Medical RAG Pipeline
Query → Hybrid Retrieval → Cross-Encoder Rerank → Gemini Answer
"""

import os
import re
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time

# ============================
# CONFIG
# ============================
DEFAULT_TOP_K = 5

# ============================
# Embeddings & Reranking
# ============================
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================
# FAISS
# ============================
import faiss

# ============================
# BM25
# ============================
from rank_bm25 import BM25Okapi

# ============================
# Gemini
# ============================
from google import genai


# =========================================================
# QUERY PROCESSOR
# =========================================================
class MedicalQueryProcessor:
    def __init__(self, embedding_model: str):
        print(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dim}")

    def extract_keywords(self, query: str) -> List[str]:
        patterns = [
            r"\b(cancer|carcinoma|tumor|neoplasm)\b",
            r"\b(ER|PR|HER2)\b",
            r"\b(stage\s*[IVX]+)\b",
            r"\b(grade\s*[123])\b",
            r"\b(lymph\s*node)\b",
        ]
        found = []
        for p in patterns:
            found.extend(re.findall(p, query, flags=re.I))
        return list(set(found))

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def process(self, query: str) -> Dict:
        return {
            "query": query,
            "keywords": self.extract_keywords(query),
            "embedding": self.embed(query),
        }


# =========================================================
# HYBRID RETRIEVER
# =========================================================
class HybridRetriever:
    def __init__(self, faiss_db_path: str):
        db = Path(faiss_db_path)

        print(f"Loading FAISS index from: {db}")
        self.index = faiss.read_index(str(db / "faiss.index"))

        with open(db / "metadata.pkl", "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        print(f"Loaded {len(self.chunks)} chunks")

        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def get_available_reports(self) -> List[str]:
        return sorted({c["filename"] for c in self.chunks})

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 40,
    ) -> List[Dict]:

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"),
            top_k,
        )

        results = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:
                results[idx] = 1 - float(dist)

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "chunk": self.chunks[idx],
                "score": score,
            }
            for idx, score in ranked
        ]


# =========================================================
# CROSS-ENCODER RERANKER
# =========================================================
class MedicalReranker:
    def __init__(self):
        print("Loading cross-encoder...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:

        if not candidates:
            return []

        pairs = [(query, c["chunk"]["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["ce_score"] = float(s)

        return sorted(
            candidates,
            key=lambda x: x["ce_score"],
            reverse=True,
        )[:top_k]


# =========================================================
# GEMINI GENERATOR
# =========================================================
class GeminiGenerator:
    def __init__(self, model_name="models/gemini-flash-lite-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"Gemini model selected: {model_name}")

    def generate(self, query: str, chunks: list) -> str:
        if not chunks:
            return "No relevant information found."

        context = ""
        for i, c in enumerate(chunks, 1):
            context += f"[{i}] {c['chunk']['text']}\n\n"

        prompt = f"""
Answer the question using ONLY the sources below.
Cite sources as [1], [2], etc.

SOURCES:
{context}

QUESTION:
{query}

ANSWER:
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text

        except genai.errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print("Rate limit hit. Retrying in 30s...")
                time.sleep(30)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text
            raise


# =========================================================
# COMPLETE RAG PIPELINE (WITH REPORT FILTERING)
# =========================================================
class CompleteRAGPipeline:
    def __init__(self, faiss_db_path: str, embedding_model: str):
        print("Initializing Complete RAG Pipeline...")
        self.query_processor = MedicalQueryProcessor(embedding_model)
        self.retriever = HybridRetriever(faiss_db_path)
        self.reranker = MedicalReranker()
        self.llm = GeminiGenerator()
        print("RAG Pipeline ready")

    def get_available_reports(self) -> List[str]:
        return self.retriever.get_available_reports()

    def ask(
        self,
        query: str,
        report_name: Optional[str] = None,
    ) -> Dict:

        processed = self.query_processor.process(query)

        candidates = self.retriever.search(
            processed["embedding"],
            query,
        )

        # ----------------------------------
        # METADATA FILTERING (DROPDOWN MODE)
        # ----------------------------------
        if report_name:
            candidates = [
                c for c in candidates
                if c["chunk"].get("filename") == report_name
            ]

            if not candidates:
                return {
                    "query": query,
                    "answer": f"No information found for report: {report_name}",
                    "timestamp": datetime.now().isoformat(),
                }

        top_chunks = self.reranker.rerank(
            query,
            candidates,
            top_k=DEFAULT_TOP_K,
        )

        answer = self.llm.generate(query, top_chunks)

        return {
            "query": query,
            "answer": answer,
            "sources": top_chunks,
            "timestamp": datetime.now().isoformat(),
        }


# =========================================================
# MAIN (TEST)
# =========================================================
def main():
    FAISS_DB = "output/biomedbert_vector_db"
    EMB_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    pipeline = CompleteRAGPipeline(FAISS_DB, EMB_MODEL)

    reports = pipeline.get_available_reports()
    print("Available reports:", reports)

    result = pipeline.ask(
        "What are the abnormal findings?",
        report_name=reports[0] if reports else None,
    )

    print(result["answer"])


if __name__ == "__main__":
    main()
