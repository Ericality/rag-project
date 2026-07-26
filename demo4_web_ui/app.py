"""
Demo 4: Streamlit Web UI for Law-RAG Hybrid Agent

Features:
- Async agent loading with progress indicator
- Upload custom PDF documents
- Switch between built-in and uploaded documents
- Chat interface with hybrid agent (vector + knowledge graph)
"""

import sys
import tempfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "demo1_vector_rag"))
sys.path.insert(0, str(PROJECT_ROOT / "demo2_knowledge_graph"))
sys.path.insert(0, str(PROJECT_ROOT / "demo3_hybrid_agent"))

import streamlit as st
from agent_graphrag import build_vector_store, build_graph, build_hybrid_agent, query_agent

# ---- Page config ----
st.set_page_config(page_title="Law-RAG", page_icon="⚖️", layout="wide")

# ---- Session state init ----
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_doc" not in st.session_state:
    st.session_state.current_doc = "built-in"
if "docs" not in st.session_state:
    st.session_state.docs = {}

# ---- Async agent loader ----
def load_agent_in_background(pdf_path, chroma_dir, cache_path, doc_label):
    """Load agent in a background thread, then update session state."""
    try:
        vectordb = build_vector_store(pdf_path=pdf_path, chroma_dir=chroma_dir)
        graph = build_graph(pdf_path=pdf_path, cache_path=cache_path)
        agent = build_hybrid_agent(vectordb, graph)
        st.session_state.agent = agent
        st.session_state.graph = graph
        st.session_state.agent_ready = True
        st.session_state.current_doc = doc_label
    except Exception as e:
        st.session_state.agent_error = str(e)
        st.session_state.agent_ready = False


def start_loading(pdf_path, chroma_dir, cache_path, doc_label):
    """Kick off background agent loading."""
    st.session_state.agent_ready = False
    st.session_state.agent_error = None
    t = threading.Thread(
        target=load_agent_in_background,
        args=(pdf_path, chroma_dir, cache_path, doc_label),
        daemon=True,
    )
    t.start()


# ---- Sidebar: Document Management ----
with st.sidebar:
    st.header("Document Management")

    # --- Built-in document ---
    st.subheader("Built-in")
    builtin_pdf = str(PROJECT_ROOT / "data" / "中华人民共和国个人信息保护法样例.pdf")
    builtin_chroma = str(PROJECT_ROOT / "chroma_db")
    builtin_cache = str(PROJECT_ROOT / "demo3_hybrid_agent" / "graph_cache.json")

    if st.button("Load Built-in Document", use_container_width=True):
        start_loading(builtin_pdf, builtin_chroma, builtin_cache, "built-in")

    st.divider()

    # --- Upload custom document ---
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"],
        help="Upload a legal document to analyze"
    )

    if uploaded_file is not None:
        doc_key = uploaded_file.name
        if doc_key not in st.session_state.docs:
            # Save uploaded file to a temp location that persists across reruns
            upload_dir = PROJECT_ROOT / "uploads"
            upload_dir.mkdir(exist_ok=True)
            saved_path = upload_dir / doc_key
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.docs[doc_key] = {
                "path": str(saved_path),
                "chroma": str(upload_dir / f"{doc_key}_chroma"),
                "cache": str(upload_dir / f"{doc_key}_graph.json"),
            }

        if st.button(f"Load '{doc_key}'", use_container_width=True):
            info = st.session_state.docs[doc_key]
            start_loading(info["path"], info["chroma"], info["cache"], doc_key)

    # --- Status display ---
    st.divider()
    if not st.session_state.agent_ready:
        if hasattr(st.session_state, "agent_error") and st.session_state.agent_error:
            st.error(f"Loading failed: {st.session_state.agent_error}")
        else:
            with st.spinner("Preparing agent..."):
                st.info("Building vector store & knowledge graph...")
    else:
        st.success("Agent ready")
        if st.session_state.graph:
            st.metric("Graph Nodes", st.session_state.graph.number_of_nodes())
            st.metric("Graph Edges", st.session_state.graph.number_of_edges())
        st.caption(f"Current: {st.session_state.current_doc}")

# ---- Main area ----
st.title("⚖️ Law-RAG: Legal Intelligence System")
st.caption("Hybrid Agent — Vector Retrieval + Knowledge Graph")

if not st.session_state.agent_ready:
    st.info("Please load a document from the sidebar to start.")
    st.subheader("Example questions you can ask:")
    examples = [
        "What principles must be followed when processing personal information?",
        "Under what circumstances is consent NOT required?",
        "What are the subcategories of personal information?",
        "What obligations does a personal information processor have?",
        "What special protections exist for minors' personal information?",
    ]
    for i, ex in enumerate(examples, 1):
        st.caption(f"{i}. {ex}")
    if hasattr(st.session_state, "agent_error") and st.session_state.agent_error:
        st.error(f"Error: {st.session_state.agent_error}")
else:
    # --- Display conversation history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    if question := st.chat_input("Ask a question about the loaded document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Agent reasoning..."):
                answer = query_agent(st.session_state.agent, question)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- Clear chat button ---
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()