"""
app.py - Streamlit entry point for the RAG PDF Summariser
---------------------------------------------------------
Run from the project root with:  streamlit run streamlit/app.py
"""

import logging
import sys
from pathlib import Path

# Make `config` and `core` importable no matter where streamlit is launched from.
# (appended, not inserted first, so the real `streamlit` package always wins
# over this project's folder that is also called `streamlit`)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st  # noqa: E402

from config.settings import CHAT_LLM_MODEL, CHUNK_OVERLAP, CHUNK_SIZE  # noqa: E402
from core.rag_chain import prompting  # noqa: E402

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUMMARY_QUESTION = (
    "Provide a professional, well-structured executive summary of this document."
)

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="RAG PDF Summariser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { background: #0f172a; }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -- Session state -------------------------------------------------------------
def _init_state():
    defaults = {
        "file_bytes": None,   # raw bytes of the processed PDF
        "summary": None,
        "qa_history": [],     # list of {"question": str, "answer": str}
        "pdf_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


@st.cache_resource
def get_chain():
    """Build the LCEL chain once and reuse it across reruns."""
    return prompting()


def run_chain(file_bytes: bytes, question: str) -> str:
    return get_chain().invoke({"file_bytes": file_bytes, "question": question})


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("API keys are read from your `.env` file.")

    st.divider()
    st.markdown("### 📂 Upload PDF")

    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        help="Only PDF files are supported.",
    )

    process_btn = st.button(
        "🚀 Process PDF", use_container_width=True, type="primary")

    if st.session_state.pdf_name:
        st.success(f"✅ **{st.session_state.pdf_name}**")

    st.divider()
    st.markdown("### 📊 Model Info")
    model_name = (
        getattr(CHAT_LLM_MODEL, "model", None)
        or getattr(CHAT_LLM_MODEL, "model_name", None)
        or type(CHAT_LLM_MODEL).__name__
    )
    st.caption(f"**LLM:** {model_name}")
    st.caption(f"**Chunk size:** {CHUNK_SIZE} chars")
    st.caption(f"**Overlap:** {CHUNK_OVERLAP} chars")

    if st.session_state.file_bytes:
        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.file_bytes = None
            st.session_state.summary = None
            st.session_state.qa_history = []
            st.session_state.pdf_name = None
            st.rerun()


# -- Main area -----------------------------------------------------------------
st.markdown("# 📄 RAG PDF Summariser")
st.markdown("Upload a PDF, generate an AI summary, then ask follow-up questions.")

# -- Processing ----------------------------------------------------------------
if process_btn:
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF file first.")
    else:
        file_bytes = uploaded_file.getvalue()
        try:
            with st.spinner("✍️ Reading, indexing and summarising the PDF …"):
                summary = run_chain(file_bytes, SUMMARY_QUESTION)
        except Exception as exc:  # surface any pipeline error in the UI
            logger.exception("Processing failed")
            st.error(f"❌ Failed to process the PDF: {exc}")
        else:
            st.session_state.file_bytes = file_bytes
            st.session_state.summary = summary
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.qa_history = []
            st.rerun()


# -- Summary panel -------------------------------------------------------------
if st.session_state.summary:
    st.markdown("---")
    st.markdown("## 📝 Document Summary")

    with st.container(border=True):
        st.markdown(st.session_state.summary)

    _, col2 = st.columns([3, 1])
    with col2:
        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.summary,
            file_name=f"{st.session_state.pdf_name}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


# -- Q&A panel -----------------------------------------------------------------
if st.session_state.file_bytes:
    st.markdown("---")
    st.markdown("## 💬 Ask Questions About the Document")

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the main conclusions of this document?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        try:
            with st.spinner("🔍 Retrieving and generating answer …"):
                answer = run_chain(
                    st.session_state.file_bytes, question.strip())
        except Exception as exc:
            logger.exception("Query failed")
            st.error(f"❌ Failed to generate an answer: {exc}")
        else:
            st.session_state.qa_history.insert(
                0, {"question": question.strip(), "answer": answer}
            )

    for item in st.session_state.qa_history:
        with st.container(border=True):
            st.markdown(f"**🙋 {item['question']}**")
            st.markdown(item["answer"])

# -- Empty state ---------------------------------------------------------------
else:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 **Upload**\n\nChoose any PDF via the sidebar")
    with col2:
        st.info("⚡ **Process**\n\nChunks are embedded into a vector store")
    with col3:
        st.info("💡 **Explore**\n\nRead the summary or ask questions")
