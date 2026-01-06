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
from typing import List, Dict

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
# Gemini (NEW SDK)
# ============================
from google import genai


# =========================================================
# QUERY PROCESSOR
# =========================================================
class MedicalQueryProcessor:
    def __init__(self, embedding_model: str):
        print(f" Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f" Embedding dimension: {self.dim}")

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
        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def process(self, query: str) -> Dict:
        return {
            "query": query,
            "keywords": self.extract_keywords(query),
            "embedding": self.embed(query),
        }


# =========================================================
# HYBRID RETRIEVER (FAISS + BM25)
# =========================================================
class HybridRetriever:
    def __init__(self, faiss_db_path: str):
        db = Path(faiss_db_path)

        print(f" Loading FAISS index from: {db}")
        self.index = faiss.read_index(str(db / "faiss.index"))

        with open(db / "metadata.pkl", "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        print(f"Loaded {len(self.chunks)} chunks")

        # Build BM25
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 40,
    ) -> List[Dict]:

        # FAISS search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"),
            top_k,
        )

        faiss_scores = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:
                faiss_scores[idx] = 1 - float(dist)

        # BM25 search
        bm25_raw = self.bm25.get_scores(query_text.lower().split())
        bm25_top = np.argsort(bm25_raw)[-top_k:]

        bm25_scores = {
            int(i): float(bm25_raw[i])
            for i in bm25_top
        }

        # Normalize & merge
        results = {}
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())

            #  Guard against all-zero BM25
        if max_bm25 > 0:
            for k, v in bm25_scores.items():
                    results[k] = results.get(k, 0) + 0.3 * (v / max_bm25)
        else:
                # All BM25 scores are zero → skip BM25 contribution
            pass


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
        print(" Loading cross-encoder...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(" Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:

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
from google import genai
import os
import time


class GeminiGenerator:
    def __init__(self, model_name="models/gemini-flash-lite-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        print(f" Gemini model selected: {model_name}")

    def generate(self, query: str, chunks: list) -> str:
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

        # Retry once on rate limit
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text

        except genai.errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(" Rate limit hit. Waiting 30 seconds and retrying...")
                time.sleep(30)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text

            raise



# =========================================================
# COMPLETE RAG PIPELINE
# =========================================================
class CompleteRAGPipeline:
    def __init__(self, faiss_db_path: str, embedding_model: str):
        print("🔧 Initializing Complete RAG Pipeline...")
        self.query_processor = MedicalQueryProcessor(embedding_model)
        self.retriever = HybridRetriever(faiss_db_path)
        self.reranker = MedicalReranker()
        self.llm = GeminiGenerator()
        print("RAG Pipeline ready")

    def ask(self, query: str) -> Dict:
        print(f"\n🔍 Query: {query}")

        processed = self.query_processor.process(query)

        candidates = self.retriever.search(
            processed["embedding"],
            query,
        )
        print(f"🔎 Retrieved {len(candidates)} candidates")

        top_chunks = self.reranker.rerank(query, candidates)
        print(f"⚡ Reranked to top {len(top_chunks)} chunks")

        answer = self.llm.generate(query, top_chunks)

        return {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        }


# =========================================================
# MAIN
# =========================================================
def main():
    print("= " * 25)
    print("MEDICAL RAG PIPELINE (GEMINI – UPDATED)")
    print("= " * 25)

    FAISS_DB = "/usr/users/3d_dimension_est/selva_sur/RAG/output/biomedbert_vector_db"
    EMB_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    pipeline = CompleteRAGPipeline(FAISS_DB, EMB_MODEL)

    queries = [
        "What are common findings in breast cancer pathology?",
        "What are typical ER/PR/HER2 receptor patterns?",
        "How is lymph node involvement assessed?",
    ]

    for q in queries:
        result = pipeline.ask(q)
        print("\n" + "=" * 80)
        print(result["answer"])
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
