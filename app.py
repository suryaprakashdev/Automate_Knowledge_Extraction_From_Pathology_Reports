#!/usr/bin/env python3
"""
Pathology RAG System - Streamlit Version
Query existing FAISS database
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add src folder
sys.path.append("src")

DB_PATH = "output/biomedbert_vector_db"

if not Path(DB_PATH).exists():
    st.error("Vector database not found. Upload output/biomedbert_vector_db.")
    st.stop()

# Import RAG pipeline
try:
    from retriever import CompleteRAGPipeline
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


# -----------------------------
# Load Pipeline (cached)
# -----------------------------

@st.cache_resource
def load_pipeline():
    # Cache busted to pick up the new ask method return dictionary
    pipeline = CompleteRAGPipeline(
        faiss_db_path=DB_PATH,
        embedding_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )

    return pipeline


pipeline = load_pipeline()


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Pathology RAG",
    layout="wide"
)

st.title("🔬 Knowledge Extraction from Pathology Report ")

st.markdown(
"""
search and question answering over pathology reports  
Vector database powered by **BiomedBERT + FAISS**
"""
)


# -----------------------------
# Session State
# -----------------------------

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("System Info")

st.sidebar.write(f"Queries: {st.session_state.query_count}")

st.sidebar.write("Embedding Model:")
st.sidebar.write("BiomedBERT")

st.sidebar.write("Vector DB:")
st.sidebar.write("FAISS")


# -----------------------------
# Query Input
# -----------------------------

st.header("🔎 Ask a Question")

question = st.text_area(
    "Enter your medical query",
    placeholder="What are common findings in breast cancer pathology?",
)

num_sources = st.slider(
    "Number of sources",
    min_value=1,
    max_value=10,
    value=5
)


# -----------------------------
# Search Button
# -----------------------------

if st.button("Search"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Running RAG pipeline..."):

            st.session_state.query_count += 1

            result = pipeline.ask(
                question,
                top_k=num_sources
            )

        answer = result["answer"]

        st.subheader("Answer")

        st.markdown(answer)


        # Metadata
        st.subheader("Query Info")

        st.write({
            "query_number": st.session_state.query_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources_used": result["num_sources"]
        })


        # Sources
        st.subheader("Sources")

        sources = result["sources"]

        if not sources:
            st.write("No sources retrieved.")

        for i, source in enumerate(sources, 1):

            chunk = source["chunk"]

            with st.expander(f"Source {i} | {chunk['filename']}"):

                st.write(chunk["text"][:600])