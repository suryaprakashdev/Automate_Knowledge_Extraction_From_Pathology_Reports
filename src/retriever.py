#!/usr/bin/env python3
"""
Complete Medical RAG Pipeline
Query → Hybrid Retrieval → Cross-Encoder Rerank → Gemini Answer
Features
--------
• FAISS semantic retrieval
• BM25 keyword retrieval
• Hybrid score fusion
• Cross-encoder reranking
• Gemini medical QA generation
• Report specific filtering (dropdown support)
"""

import os
import re
import pickle
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import faiss
from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai


# ============================================
# CONFIG
# ============================================

DEFAULT_TOP_K = 5


# ============================================
# QUERY PROCESSOR
# ============================================

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


# ============================================
# HYBRID RETRIEVER
# ============================================

class HybridRetriever:

    def __init__(self, faiss_db_path: str):

        db = Path(faiss_db_path)

        print(f"Loading FAISS index from: {db}")

        self.index = faiss.read_index(str(db / "faiss.index"))

        with open(db / "metadata.pkl", "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]

        print(f"Loaded {len(self.chunks)} chunks")

        tokenized = [
            c["text"].lower().split()
            for c in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    def get_available_reports(self) -> List[str]:

        return sorted({
            c["filename"]
            for c in self.chunks
        })

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 40
    ) -> List[Dict]:

        # -----------------------------------
        # FAISS SEMANTIC SEARCH
        # -----------------------------------

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"),
            top_k
        )

        faiss_scores = {}

        for idx, dist in zip(indices[0], distances[0]):

            if idx >= 0:
                faiss_scores[int(idx)] = float(dist)

        if not faiss_scores:
            return []

        # -----------------------------------
        # BM25 KEYWORD SEARCH
        # -----------------------------------

        tokens = query_text.lower().split()

        bm25_raw = self.bm25.get_scores(tokens)

        bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1.0

        bm25_scores = {
            i: bm25_raw[i] / bm25_max
            for i in faiss_scores.keys()
        }

        # -----------------------------------
        # HYBRID SCORE FUSION
        # -----------------------------------

        fused_scores = {}

        for idx in faiss_scores.keys():

            faiss_score = faiss_scores.get(idx, 0)

            bm25_score = bm25_scores.get(idx, 0)

            fused_scores[idx] = (
                0.7 * faiss_score +
                0.3 * bm25_score
            )

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for idx, score in ranked:

            results.append(
                {
                    "chunk": self.chunks[idx],
                    "score": score
                }
            )

        return results


# ============================================
# CROSS ENCODER RERANKER
# ============================================

class MedicalReranker:

    def __init__(self):

        print("Loading cross-encoder reranker...")

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("Cross-encoder ready")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:

        if not candidates:
            return []

        pairs = [
            (query, c["chunk"]["text"])
            for c in candidates
        ]

        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["ce_score"] = float(s)

        ranked = sorted(
            candidates,
            key=lambda x: x["ce_score"],
            reverse=True
        )

        return ranked[:top_k]


# ============================================
# GEMINI GENERATOR
# ============================================

class GeminiGenerator:

    def __init__(
        self,
        model_name="models/gemini-flash-lite-latest"
    ):

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        self.client = genai.Client(api_key=api_key)

        self.model_name = model_name

        print(f"Using Gemini model: {model_name}")

    def generate(
        self,
        query: str,
        chunks: List[Dict]
    ) -> str:

        if not chunks:
            return "No relevant information found."

        context = ""

        for i, c in enumerate(chunks, 1):
            chunk_text = c['chunk'].get('text', '')
            
            # If the database was built without the original text files, it defaults to "Document: <filename>"
            # We can still provide the extracted entities to Gemini so it has context!
            if chunk_text.startswith("Document:") and "entities" in c['chunk']:
                entities = [e.get("text", "") for e in c['chunk']["entities"]]
                chunk_text += "\nExtracted Medical Entities: " + ", ".join(entities)

            context += f"[{i}] {chunk_text}\n\n"

        prompt = f"""
You are an expert medical assistant. Answer the medical question using ONLY the provided text or entities from the pathology documents below.
Treat the provided information as your complete source material. Do not state that you cannot access files, as the contents/entities are provided directly below.
Cite your sources in your text using their corresponding numbers like [1], [2], etc.

--- PROVIDED DOCUMENT CONTENTS / ENTITIES ---
{context}

--- QUESTION ---
{query}

--- ANSWER ---
"""

        try:

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            return response.text

        except Exception as e:

            if "RESOURCE_EXHAUSTED" in str(e):

                print("Rate limit reached. Waiting 30 seconds...")

                time.sleep(30)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )

                return response.text

            raise


# ============================================
# COMPLETE RAG PIPELINE
# ============================================

class CompleteRAGPipeline:

    def __init__(
        self,
        faiss_db_path: str,
        embedding_model: str
    ):

        print("\nInitializing Medical RAG Pipeline\n")

        self.query_processor = MedicalQueryProcessor(
            embedding_model
        )

        self.retriever = HybridRetriever(
            faiss_db_path
        )

        self.reranker = MedicalReranker()

        self.llm = GeminiGenerator()

        print("\nPipeline ready\n")

    def get_available_reports(self) -> List[str]:
        return self.retriever.get_available_reports()

    def ask(
        self,
        query: str,
        report_name: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:

        processed = self.query_processor.process(query)

        candidates = self.retriever.search(
            processed["embedding"],
            query
        )

        # ----------------------------------
        # REPORT FILTERING
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
                    "sources": [],
                    "num_sources": 0
                }

        # ----------------------------------
        # RERANK
        # ----------------------------------

        top_chunks = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k
        )

        # ----------------------------------
        # GENERATE ANSWER
        # ----------------------------------

        answer = self.llm.generate(
            query,
            top_chunks
        )

        return {
            "query": query,
            "answer": answer,
            "sources": top_chunks,
            "timestamp": datetime.now().isoformat(),
            "num_sources": len(top_chunks)
        }


# ============================================
# MAIN TEST
# ============================================

def main():

    FAISS_DB = "output/biomedbert_vector_db"

    EMB_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    pipeline = CompleteRAGPipeline(
        FAISS_DB,
        EMB_MODEL
    )

    reports = pipeline.get_available_reports()

    print("\nAvailable reports:")
    print(reports)

    result = pipeline.ask(
        "What abnormal findings are present?",
        report_name=reports[0] if reports else None
    )

    print("\nAnswer:\n")
    print(result["answer"])


if __name__ == "__main__":
    main()