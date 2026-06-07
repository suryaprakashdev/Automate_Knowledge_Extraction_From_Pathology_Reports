#!/usr/bin/env python3
"""
Pathology RAG System — Streamlit UI
Two pages:
  1. Upload & Query  — upload a PDF, view it, ask questions scoped to that file
  2. Global Search   — ask questions across the entire vector database
"""

import os
import sys
import base64
from pathlib import Path
from datetime import datetime
import streamlit as st

# ── Force CPU ──────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Source path ────────────────────────────────────────────────────────────────
sys.path.append("src")

DB_PATH = "output/biomedbert_vector_db"
EMBEDDING_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# ── Validate database ──────────────────────────────────────────────────────────
if not Path(DB_PATH).exists():
    st.error("⚠️ Vector database not found at `output/biomedbert_vector_db/`. "
             "Please build it first.")
    st.stop()

# ── Imports ────────────────────────────────────────────────────────────────────
try:
    from retriever import CompleteRAGPipeline
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pathology RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ── */
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }
/* ── Answer box ── */
.answer-box {
    background: #0f1117;
    border: 1px solid #2d3748;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.5rem;
    color: #e2e8f0;
    font-size: 0.97rem;
    line-height: 1.7;
}
/* ── Source card ── */
.source-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.87rem;
    color: #94a3b8;
}
.source-card strong { color: #7dd3fc; }
/* ── Upload status ── */
.upload-success {
    background: #052e16;
    border: 1px solid #15803d;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    color: #4ade80;
    font-size: 0.9rem;
}
/* ── Metric pills ── */
.metric-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "query_count":        0,
    "pdf_bytes":          None,   # raw bytes of the currently-loaded PDF
    "pdf_name":           None,   # display name  (e.g. "report.pdf")
    "pdf_stem":           None,   # chunk filename key (e.g. "report")
    "upload_stats":       None,
    "doc_answer":         None,
    "doc_sources":        None,
    "global_answer":      None,
    "global_sources":     None,
    "db_update_time":     0.0,    # Cache-busting trigger timestamp
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline(db_update_time: float):
    """Loads or reloads the entire pipeline cleanly whenever db_update_time updates."""
    return CompleteRAGPipeline(
        faiss_db_path=DB_PATH,
        embedding_model=EMBEDDING_MODEL,
    )

# Always fetch pipeline using the state tracker dependency
pipeline = load_pipeline(st.session_state.db_update_time)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_pdf_viewer(pdf_name: str, height: int = 820):
    """
    Instead of Base64, we point the iframe to a local static path 
    or use the raw uploaded path if using a component.
    """
    # Look into the uploaded reports directory where process_upload saves it
    pdf_path = Path("uploaded_reports") / pdf_name
    
    if not pdf_path.exists():
        st.error("Physical PDF file not found on disk.")
        return

    # To bypass Chrome's nested iframe blocks on Hugging Face,
    # the cleanest approach is reading the bytes and using streamlit-pdf-viewer,
    # OR utilizing an embedded object tag with a local stream.
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    
    # Using <object> instead of <iframe> sometimes tricks Chrome's iframe blocker,
    # but the ultimate fix for HF Spaces is serving via an actual endpoint.
    pdf_display = f"""
    <object data="data:application/pdf;base64,{b64}" type="application/pdf" width="100%" height="{height}px">
        <embed src="data:application/pdf;base64,{b64}" type="application/pdf" />
    </object>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

def render_answer(answer: str):
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

def render_sources(sources: list):
    if not sources:
        st.caption("No sources retrieved.")
        return
    for i, src in enumerate(sources, 1):
        chunk = src["chunk"]
        score = src.get("ce_score", src.get("score", 0))
        text_preview = chunk.get("text", "")[:400].replace("\n", " ")
        st.markdown(f"""
        <div class="source-card">
            <strong>Source {i} — {chunk.get('filename', 'unknown')}</strong>
            &nbsp;<span style="color:#475569">score {score:.3f}</span><br/>
            <span>{text_preview}…</span>
        </div>
        """, unsafe_allow_html=True)

def process_upload(uploaded_file) -> bool:
    """Save PDF, run OCR + embedding, update FAISS. Returns stats dict."""
    from document_processor import DynamicRAGUpdater
    upload_dir = Path("uploaded_reports")
    upload_dir.mkdir(exist_ok=True)
    pdf_path = upload_dir / uploaded_file.name
    pdf_bytes = uploaded_file.getbuffer()
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    updater = DynamicRAGUpdater(
        vector_db_path=DB_PATH,
        upload_dir=str(upload_dir),
    )
    stats = updater.process_and_add_pdf(str(pdf_path))
    
    # Persist tracking metrics
    st.session_state.upload_stats = stats
    st.session_state.doc_answer  = None
    st.session_state.doc_sources = None
    return stats

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — navigation + persistent PDF info
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Pathology RAG")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📄  Upload & Query", "🌐  Global Search"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**System**")
    st.caption(f"Embedding: BiomedBERT")
    st.caption(f"Index: FAISS + BM25 Hybrid")
    st.caption(f"Reranker: CrossEncoder")
    st.caption(f"LLM: Gemini Flash Lite")
    st.divider()
    st.caption(f"Queries this session: **{st.session_state.query_count}**")
    if st.session_state.pdf_name:
        st.divider()
        st.caption("**Active document**")
        st.success(f"📎 {st.session_state.pdf_name}")
        s = st.session_state.upload_stats
        if s:
            st.caption(f"Chunks: {s['num_chunks']} · Vectors: {s['vectors_added']}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Upload & Query
# ══════════════════════════════════════════════════════════════════════════════
if page == "📄  Upload & Query":
    st.title("📄 Upload & Query")
    st.caption("Upload a pathology PDF, view it, and ask questions scoped to that document.")

    # FIX 1: Intercept the uploaded file early so state updates BEFORE layout columns render
    uploaded_file = st.file_uploader(
        "Select a PDF pathology report",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            st.session_state.pdf_bytes = bytes(uploaded_file.getbuffer())
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_stem = Path(uploaded_file.name).stem
    else:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_name = None
        st.session_state.pdf_stem = None

    # ── Layout: viewer (left) | controls (right) ──────────────────────────────
    col_viewer, col_controls = st.columns([1.1, 1], gap="large")

    # ── LEFT — PDF viewer ──────────────────────────────────────────────────────
    with col_viewer:
        st.markdown("#### Document Viewer")
        if st.session_state.pdf_bytes:
            render_pdf_viewer(st.session_state.pdf_bytes, height=820)
        else:
            st.markdown(
                """
                <div style="
                    height:820px;
                    border:2px dashed #2d3748;
                    border-radius:10px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:#4b5563;
                    font-size:1rem;
                ">
                    📂 Upload a PDF to preview it here
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── RIGHT — upload + query ─────────────────────────────────────────────────
    with col_controls:
        st.markdown("#### Upload Report")
        
        # Explicit process action button
        process_btn = st.button(
            "⚙️ Process & Index Document",
            use_container_width=True,
            disabled=(uploaded_file is None),
            key="process_btn",
        )
        
        if process_btn and uploaded_file is not None:
            with st.spinner(f"🔄 Processing `{uploaded_file.name}` — OCR + embedding + indexing…"):
                try:
                    stats = process_upload(uploaded_file)
                    
                    # FIX 2: Alter the timestamp parameter. This tells @st.cache_resource 
                    # to run load_pipeline() again, forcing it to read the new index files off disk.
                    st.session_state.db_update_time = datetime.now().timestamp()
                    st.rerun()
                    
                except Exception as e:
                    import traceback
                    st.error(f"❌ Processing failed: {e}")
                    st.code(traceback.format_exc(), language="text")

        # Show success toast if file metrics match current session active item
        if st.session_state.upload_stats and st.session_state.pdf_name == uploaded_file.name if uploaded_file else False:
            s = st.session_state.upload_stats
            st.markdown(
                f"""
                <div class="upload-success">
                    ✅ <strong>{st.session_state.pdf_name}</strong> processed and indexed successfully<br/>
                    Chunks: {s['num_chunks']} · Vectors added: {s['vectors_added']} · Time: {s['processing_time_seconds']:.1f}s
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Query section (only active when a PDF is loaded) ───────────────────
        st.markdown("#### Ask About This Document")
        if not st.session_state.pdf_stem:
            st.info("⬆️ Upload a PDF above to enable document-specific search.")
        else:
            st.caption(f"Searching within: **{st.session_state.pdf_name}**")
            doc_question = st.text_area(
                "Your question",
                placeholder="e.g. What are the key abnormal findings in this report?",
                height=110,
                key="doc_question",
                label_visibility="collapsed",
            )
            num_sources_doc = st.slider(
                "Sources to retrieve",
                min_value=1, max_value=10, value=5,
                key="doc_sources_slider",
            )
            
            if st.button("🔍 Ask Document", use_container_width=True, key="doc_ask_btn"):
                if not doc_question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Running RAG pipeline…"):
                        st.session_state.query_count += 1
                        
                        # CRITICAL CRUMB CHECK: Ensure pipeline.ask uses full name or stem 
                        # matching how your metadata parser formats the 'filename' key.
                        result = pipeline.ask(
                            doc_question,
                            report_name=st.session_state.pdf_stem, 
                            top_k=num_sources_doc,
                        )
                        st.session_state.doc_answer  = result["answer"]
                        st.session_state.doc_sources = result["sources"]

            if st.session_state.doc_answer:
                st.markdown("**Answer**")
                render_answer(st.session_state.doc_answer)
                with st.expander("📚 Sources", expanded=False):
                    render_sources(st.session_state.doc_sources)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Global Search
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐  Global Search":
    st.title("🌐 Global Search")
    st.caption("Ask questions across **all** indexed pathology reports.")
    
    try:
        available = pipeline.get_available_reports()
        pills = "".join(f'<span class="metric-pill">📄 {r}</span>' for r in available[:20])
        suffix = f'<span class="metric-pill">+{len(available)-20} more</span>' if len(available) > 20 else ""
        st.markdown(f"**{len(available)} reports indexed**<br/>{pills}{suffix}", unsafe_allow_html=True)
    except Exception:
        pass

    st.divider()

    global_question = st.text_area(
        "Your question",
        placeholder="e.g. What are the most common ER/PR/HER2 receptor patterns found across reports?",
        height=130,
        label_visibility="collapsed",
        key="global_question",
    )
    col_slider, col_btn = st.columns([3, 1], gap="small")
    with col_slider:
        num_sources_global = st.slider(
            "Sources to retrieve",
            min_value=1, max_value=10, value=5,
            key="global_sources_slider",
        )
    with col_btn:
        st.markdown("<br/>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Search All", use_container_width=True, key="global_search_btn")

    if search_clicked:
        if not global_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching across all documents…"):
                st.session_state.query_count += 1
                result = pipeline.ask(
                    global_question,
                    top_k=num_sources_global,
                )
                st.session_state.global_answer  = result["answer"]
                st.session_state.global_sources = result["sources"]
                
                st.markdown(
                    f"""
                    <div style="margin:0.5rem 0 0.25rem;">
                        <span class="metric-pill">Query #{st.session_state.query_count}</span>
                        <span class="metric-pill">Sources used: {result['num_sources']}</span>
                        <span class="metric-pill">{datetime.now().strftime('%H:%M:%S')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if st.session_state.global_answer:
        st.markdown("**Answer**")
        render_answer(st.session_state.global_answer)
        st.markdown("<br/>", unsafe_allow_html=True)
        with st.expander("📚 Retrieved Sources", expanded=True):
            render_sources(st.session_state.global_sources)