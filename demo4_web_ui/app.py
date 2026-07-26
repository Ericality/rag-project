"""
Demo 4: Streamlit Web UI for Law-RAG Hybrid Agent

Features:
- Progress indicator during agent loading
- Upload custom PDF documents
- Switch between built-in and uploaded documents
- Chat interface with hybrid agent (vector + knowledge graph)
"""

import sys
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
    st.session_state.current_doc = None
if "docs" not in st.session_state:
    st.session_state.docs = {}


def load_agent(pdf_path, chroma_dir, cache_path, doc_label):
    """Synchronous agent builder. Called inside st.spinner."""
    vectordb = build_vector_store(pdf_path=pdf_path, chroma_dir=chroma_dir)
    graph = build_graph(pdf_path=pdf_path, cache_path=cache_path)
    agent = build_hybrid_agent(vectordb, graph)
    st.session_state.agent = agent
    st.session_state.graph = graph
    st.session_state.agent_ready = True
    st.session_state.current_doc = doc_label


# ---- Sidebar: Document Management ----
with st.sidebar:
    st.header("Document Management")

    # --- Built-in document ---
    st.subheader("Built-in")
    builtin_pdf = str(PROJECT_ROOT / "data" / "中华人民共和国个人信息保护法样例.pdf")
    builtin_chroma = str(PROJECT_ROOT / "chroma_db")
    builtin_cache = str(PROJECT_ROOT / "demo3_hybrid_agent" / "graph_cache.json")

    if st.button("Load Built-in Document", use_container_width=True):
        with st.spinner("Building vector store & knowledge graph..."):
            try:
                load_agent(builtin_pdf, builtin_chroma, builtin_cache, "Built-in Document")
            except Exception as e:
                st.error(f"Failed to load: {e}")

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
            with st.spinner(f"Building vector store & knowledge graph for '{doc_key}'..."):
                try:
                    load_agent(info["path"], info["chroma"], info["cache"], doc_key)
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    # --- Status display ---
    st.divider()
    if st.session_state.agent_ready and st.session_state.graph:
        st.success("Agent ready")
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
else:
    # Example questions dropdown
    example_questions = [
        "What principles must be followed when processing personal information?",
        "Under what circumstances is consent NOT required?",
        "What are the subcategories of personal information?",
        "What obligations does a personal information processor have?",
        "What special protections exist for minors' personal information?",
    ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick-select radio
    example_questions_cn = [
        "处理个人信息需要满足哪些原则？",
        "在什么情况下处理个人信息不需要取得个人同意？",
        "个人信息包含哪些子类别？",
        "个人信息处理者有哪些义务？",
        "未成年人个人信息保护有什么特别规定？",
    ]
    selected = st.radio(
        "快捷问题（点击即可发送）：",
        [""] + example_questions_cn,
        index=0,
    )
    question = None
    if selected:
        question = selected
    else:
        question = st.chat_input("或在下方输入你的问题…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Agent reasoning..."):
                answer = query_agent(st.session_state.agent, question)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
