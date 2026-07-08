#!/usr/bin/env python3
"""
Pathology RAG System — Streamlit UI
Two pages:
  1. Upload & Query  — upload a PDF, view it, ask questions scoped to that file,
                        with matching lines highlighted directly on the page.
  2. Global Search   — ask questions across the entire vector database,
                        optionally scoped to a subset of documents.
"""

import os
import sys
import base64
from pathlib import Path
from datetime import datetime
import streamlit as st
import fitz  # PyMuPDF

# ── Force CPU ──────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Source / UI paths ─────────────────────────────────────────────────────────
sys.path.append("src")
sys.path.append(str(Path(__file__).parent))
from ui.render import load_css, render as render_ui

DB_PATH = "output/biomedbert_vector_db"
EMBEDDING_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
UPLOAD_DIR = "uploaded_reports"
HIGHLIGHT_COLOR = (1.0, 0.85, 0.25)  # RGB 0-1, warm yellow

# ── Validate database ──────────────────────────────────────────────────────────
if not Path(DB_PATH).exists():
    st.error("⚠️ Vector database not found at `output/biomedbert_vector_db/`. "
             "Please build it first.")
    st.stop()

# ── Imports ────────────────────────────────────────────────────────────────────
try:
    from retriever import CompleteRAGPipeline
    from document_processor import DynamicRAGUpdater
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

# ── Custom CSS (ui/styles.css) ────────────────────────────────────────────────
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "query_count":        0,
    "pdf_bytes":          None,   # raw bytes of the currently-loaded PDF
    "pdf_name":           None,   # display name  (e.g. "report.pdf")
    "pdf_stem":           None,   # chunk filename key (e.g. "report")
    "pdf_num_pages":      1,
    "pdf_page":           1,      # currently displayed page (1-indexed)
    "pdf_highlight_bboxes": None, # bboxes to highlight on pdf_page, or None
    "upload_stats":       None,
    "doc_answer":         None,
    "doc_sources":        None,
    "global_answer":      None,
    "global_sources":     None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
#
# Model loads are expensive (BiomedBERT, cross-encoder, PaddleOCR). Everything
# here is built exactly once for the life of the app process and reused across
# reruns/uploads. New documents are picked up via pipeline.reload_index(),
# which only re-reads the FAISS index + metadata — it never reloads a model.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline():
    return CompleteRAGPipeline(
        faiss_db_path=DB_PATH,
        embedding_model=EMBEDDING_MODEL,
    )

@st.cache_resource
def load_ocr_engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        cpu_threads=4,
        enable_mkldnn=False,
    )

@st.cache_resource
def load_updater(_pipeline, _ocr_engine):
    """Reuses the pipeline's already-loaded BiomedBERT embedder for document
    embedding too, so only one copy of that model ever sits in memory."""
    return DynamicRAGUpdater(
        vector_db_path=DB_PATH,
        upload_dir=UPLOAD_DIR,
        ocr_engine=_ocr_engine,
        embedder=_pipeline.query_processor.model,
    )

def get_updater():
    """Lazily builds (once) and returns the document updater. Only called when
    a document is actually being processed, so pages that don't touch OCR
    (e.g. Global Search) never pay PaddleOCR's startup cost."""
    ocr_engine = load_ocr_engine()
    return load_updater(pipeline, ocr_engine)

pipeline = load_pipeline()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_num_pages(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n

def render_pdf_page_image(pdf_path: Path, page_num: int, bboxes=None, dpi: int = 150) -> bytes:
    """Renders a single PDF page to PNG bytes, optionally drawing highlight
    rectangles for the given bboxes (in PDF point coordinates)."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)

    if bboxes:
        for bbox in bboxes:
            rect = fitz.Rect(*bbox)
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=HIGHLIGHT_COLOR)
            annot.update()

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes

def render_pdf_object_embed(pdf_path: Path, height: int = 820):
    """Raw browser-native PDF embed — fallback for documents that predate
    bbox/page tracking and therefore can't be highlighted."""
    if not pdf_path.exists():
        st.error(f"Physical PDF file not found at {pdf_path}")
        return
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(render_ui("pdf_object_embed.html", b64=b64, height=height), unsafe_allow_html=True)

def render_answer(answer: str):
    st.markdown(render_ui("answer_box.html", answer=answer), unsafe_allow_html=True)

def render_sources(sources: list):
    if not sources:
        st.caption("No sources retrieved.")
        return
    for i, src in enumerate(sources, 1):
        chunk = src["chunk"]
        score = src.get("ce_score", src.get("score", 0))
        text_preview = chunk.get("text", "")[:400].replace("\n", " ")
        st.markdown(render_ui(
            "source_card.html",
            index=i,
            filename=chunk.get("filename", "unknown"),
            page=chunk.get("page"),
            score=score,
            text_preview=text_preview,
        ), unsafe_allow_html=True)

def process_upload(uploaded_file) -> dict:
    """Save PDF, run OCR + embedding via the cached updater, update FAISS,
    then cheaply refresh the query pipeline's view of the index."""
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    pdf_path = upload_dir / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    stats = get_updater().process_and_add_pdf(str(pdf_path))
    pipeline.reload_index()

    st.session_state.upload_stats = stats
    st.session_state.doc_answer = None
    st.session_state.doc_sources = None
    return stats

def set_active_pdf(pdf_name: str, pdf_path: Path):
    st.session_state.pdf_name = pdf_name
    st.session_state.pdf_stem = Path(pdf_name).stem
    st.session_state.pdf_num_pages = get_num_pages(pdf_path)
    st.session_state.pdf_page = 1
    st.session_state.pdf_highlight_bboxes = None

def focus_on_sources(sources: list):
    """Jumps the viewer to the top-ranked source's page and highlights every
    line on that page belonging to any of the top sources. Falls back to no
    highlight if this document predates page/bbox tracking."""
    if not sources:
        return
    top_page = sources[0]["chunk"].get("page")
    if not top_page:
        st.session_state.pdf_highlight_bboxes = None
        return
    bboxes = []
    for src in sources:
        chunk = src["chunk"]
        if chunk.get("page") == top_page:
            bboxes.extend(chunk.get("line_bboxes", []))
    st.session_state.pdf_page = top_page
    st.session_state.pdf_highlight_bboxes = bboxes or None

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
    st.caption("Embedding: BiomedBERT")
    st.caption("Index: FAISS + BM25 Hybrid")
    st.caption("Reranker: CrossEncoder")
    st.caption("LLM: Gemini Flash Lite")
    try:
        n_docs = len(pipeline.get_available_reports())
        st.caption(f"Documents indexed: **{n_docs}**")
    except Exception:
        pass
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
    st.markdown(render_ui(
        "page_header.html",
        icon="📄", title="Upload & Query",
        subtitle="Upload a pathology PDF, view it, and ask questions scoped to that "
                 "document — matching lines are highlighted directly on the page.",
    ), unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select a PDF pathology report",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            pdf_path = Path(UPLOAD_DIR) / uploaded_file.name
            Path(UPLOAD_DIR).mkdir(exist_ok=True)
            # Stage the raw bytes to disk immediately so the viewer can render
            # it even before "Process & Index" is clicked.
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            set_active_pdf(uploaded_file.name, pdf_path)
    else:
        st.session_state.pdf_name = None
        st.session_state.pdf_stem = None

    col_viewer, col_controls = st.columns([1.1, 1], gap="large")

    # ── LEFT — PDF viewer ──────────────────────────────────────────────────
    with col_viewer:
        st.markdown(render_ui("panel_title.html", title="Document Viewer"), unsafe_allow_html=True)
        if st.session_state.pdf_name:
            pdf_path = Path(UPLOAD_DIR) / st.session_state.pdf_name
            has_highlight = bool(st.session_state.pdf_highlight_bboxes)

            try:
                img_bytes = render_pdf_page_image(
                    pdf_path,
                    st.session_state.pdf_page,
                    bboxes=st.session_state.pdf_highlight_bboxes,
                )
                st.image(img_bytes, use_container_width=True)

                nav_l, nav_c, nav_r = st.columns([1, 2, 1])
                with nav_l:
                    if st.button("◀ Prev", disabled=st.session_state.pdf_page <= 1, use_container_width=True):
                        st.session_state.pdf_page -= 1
                        st.session_state.pdf_highlight_bboxes = None
                        st.rerun()
                with nav_c:
                    st.markdown(render_ui(
                        "pdf_navbar.html",
                        current_page=st.session_state.pdf_page,
                        num_pages=st.session_state.pdf_num_pages,
                    ), unsafe_allow_html=True)
                with nav_r:
                    if st.button("Next ▶", disabled=st.session_state.pdf_page >= st.session_state.pdf_num_pages, use_container_width=True):
                        st.session_state.pdf_page += 1
                        st.session_state.pdf_highlight_bboxes = None
                        st.rerun()

                if has_highlight:
                    st.markdown(render_ui("highlight_note.html"), unsafe_allow_html=True)
            except Exception:
                render_pdf_object_embed(pdf_path, height=820)
        else:
            st.markdown(render_ui(
                "empty_state.html", icon="📂", text="Upload a PDF to preview it here",
            ), unsafe_allow_html=True)

    # ── RIGHT — upload + query ─────────────────────────────────────────────
    with col_controls:
        st.markdown(render_ui("panel_title.html", title="Upload Report"), unsafe_allow_html=True)

        process_btn = st.button(
            "⚙️ Process & Index Document",
            use_container_width=True,
            disabled=(uploaded_file is None),
            key="process_btn",
        )

        if process_btn and uploaded_file is not None:
            with st.spinner(f"🔄 Processing `{uploaded_file.name}` — OCR → embedding → indexing…"):
                try:
                    process_upload(uploaded_file)
                    pdf_path = Path(UPLOAD_DIR) / uploaded_file.name
                    set_active_pdf(uploaded_file.name, pdf_path)
                    st.rerun()
                except Exception as e:
                    import traceback
                    st.error(f"❌ Processing failed: {e}")
                    st.code(traceback.format_exc(), language="text")

        if st.session_state.upload_stats and uploaded_file and st.session_state.pdf_name == uploaded_file.name:
            s = st.session_state.upload_stats
            st.markdown(render_ui(
                "upload_success.html",
                pdf_name=st.session_state.pdf_name,
                num_chunks=s["num_chunks"],
                vectors_added=s["vectors_added"],
                processing_time=s["processing_time_seconds"],
            ), unsafe_allow_html=True)

        st.divider()

        st.markdown(render_ui("panel_title.html", title="Ask About This Document"), unsafe_allow_html=True)
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
                        result = pipeline.ask(
                            doc_question,
                            report_name=st.session_state.pdf_stem,
                            top_k=num_sources_doc,
                        )
                        st.session_state.doc_answer = result["answer"]
                        st.session_state.doc_sources = result["sources"]
                        focus_on_sources(result["sources"])
                    st.rerun()

            if st.session_state.doc_answer:
                st.markdown("**Answer**")
                render_answer(st.session_state.doc_answer)
                with st.expander("📚 Sources", expanded=False):
                    render_sources(st.session_state.doc_sources)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Global Search
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐  Global Search":
    st.markdown(render_ui(
        "page_header.html",
        icon="🌐", title="Global Search",
        subtitle="Ask questions across any number of indexed pathology reports.",
    ), unsafe_allow_html=True)

    try:
        available = pipeline.get_available_reports()
    except Exception:
        available = []

    if available:
        st.markdown(render_ui(
            "report_pills.html",
            total=len(available),
            reports=available[:20],
            remaining=max(0, len(available) - 20),
        ), unsafe_allow_html=True)

    st.divider()

    doc_filter = st.multiselect(
        "Scope to specific documents (optional)",
        options=available,
        default=[],
        placeholder="Leave empty to search all indexed documents",
        key="global_doc_filter",
    )

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
        search_clicked = st.button("🔍 Search", use_container_width=True, key="global_search_btn")

    if search_clicked:
        if not global_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching across documents…"):
                st.session_state.query_count += 1
                result = pipeline.ask(
                    global_question,
                    report_names=doc_filter or None,
                    top_k=num_sources_global,
                )
                st.session_state.global_answer = result["answer"]
                st.session_state.global_sources = result["sources"]

                st.markdown(render_ui(
                    "query_metrics.html",
                    query_count=st.session_state.query_count,
                    num_sources=result["num_sources"],
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                ), unsafe_allow_html=True)

    if st.session_state.global_answer:
        st.markdown("**Answer**")
        render_answer(st.session_state.global_answer)
        st.markdown("<br/>", unsafe_allow_html=True)
        with st.expander("📚 Retrieved Sources", expanded=True):
            render_sources(st.session_state.global_sources)
