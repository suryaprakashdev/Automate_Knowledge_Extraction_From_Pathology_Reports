import streamlit as st
import sys
from pathlib import Path

# -------------------------------------------------
# Add src/ to Python path
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from retriever import CompleteRAGPipeline

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
FAISS_DB = str(BASE_DIR / "output" / "biomedbert_vector_db")
EMB_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Medical RAG Assistant")
st.markdown(
    """
    Ask questions about **pathology reports**.  
    Answers are generated **only from indexed medical documents**.
    """
)

# -------------------------------------------------
# LOAD PIPELINE (cached)
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    return CompleteRAGPipeline(
        faiss_db_path=FAISS_DB,
        embedding_model=EMB_MODEL,
    )

pipeline = load_pipeline()

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
query = st.text_input(
    "Enter your medical question:",
    placeholder="e.g., How is lymph node involvement assessed?",
)

ask_button = st.button("🔍 Ask")

# -------------------------------------------------
# RUN QUERY
# -------------------------------------------------
if ask_button and query.strip():
    with st.spinner("🔎 Retrieving relevant documents and generating answer..."):
        result = pipeline.ask(query)

    st.markdown("## ✅ Answer")
    st.write(result["answer"])

elif ask_button:
    st.warning("Please enter a question.")
