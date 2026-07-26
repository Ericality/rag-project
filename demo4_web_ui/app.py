"""
Demo 4: Streamlit Web UI for Law-RAG Hybrid Agent

Features:
- Language toggle (EN / 中文)
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
if "lang" not in st.session_state:
    st.session_state.lang = "en"
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

# ---- i18n ----
T = {
    "en": {
        "lang_label": "🌐 Language",
        "doc_mgmt": "📁 Document Management",
        "builtin": "Built-in",
        "load_builtin": "Load Built-in Document",
        "building": "Building vector store & knowledge graph...",
        "load_failed": "Failed to load",
        "upload": "Upload Document",
        "upload_prompt": "Choose a PDF file",
        "upload_help": "Upload a legal document to analyze",
        "load_upload": "Load '{name}'",
        "building_for": "Building vector store & knowledge graph for '{name}'...",
        "agent_ready": "Agent ready",
        "graph_nodes": "Graph Nodes",
        "graph_edges": "Graph Edges",
        "current": "Current",
        "title": "⚖️ Law-RAG: Legal Intelligence System",
        "subtitle": "Hybrid Agent — Vector Retrieval + Knowledge Graph",
        "please_load": "Please load a document from the sidebar to start.",
        "example_label": "Example questions:",
        "quick_label": "Quick questions (select to send):",
        "quick_placeholder": "Click to choose a question, or type below…",
        "chat_placeholder": "or type your question here…",
        "reasoning": "Agent reasoning...",
        "clear_chat": "Clear Chat",
        "examples": [
            "What principles must be followed when processing personal information?",
            "Under what circumstances is consent NOT required?",
            "What behaviors are prohibited for personal information processors?",
            "What are the specific requirements for consent?",
            "What are the lawful, legitimate, and necessary principles?",
        ],
    },
    "cn": {
        "lang_label": "🌐 语言",
        "doc_mgmt": "📁 文档管理",
        "builtin": "内置文档",
        "load_builtin": "加载内置文档",
        "building": "正在构建向量库和知识图谱...",
        "load_failed": "加载失败",
        "upload": "上传文档",
        "upload_prompt": "选择 PDF 文件",
        "upload_help": "上传法律文档进行分析",
        "load_upload": "加载「{name}」",
        "building_for": "正在为「{name}」构建向量库和知识图谱...",
        "agent_ready": "Agent 就绪",
        "graph_nodes": "图谱节点数",
        "graph_edges": "图谱边数",
        "current": "当前文档",
        "title": "⚖️ Law-RAG: 法律智能问答系统",
        "subtitle": "混合 Agent — 向量检索 + 知识图谱",
        "please_load": "请从侧边栏加载文档以开始使用。",
        "example_label": "示例问题：",
        "quick_label": "快捷问题（选择后自动发送）：",
        "quick_placeholder": "点击选择一个问题，或直接在下方输入…",
        "chat_placeholder": "或在此输入你的问题…",
        "reasoning": "Agent 推理中...",
        "clear_chat": "清空对话",
        "examples": [
            "处理个人信息需要遵循哪些原则？",
            "在什么情况下处理个人信息不需要取得个人同意？",
            "个人信息处理者有哪些禁止行为？",
            "同意有哪些具体要求？",
            "什么是合法、正当、必要原则？",
        ],
    },
}


def t(key, **kwargs):
    """Look up a translated string, with optional formatting."""
    text = T[st.session_state.lang].get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# ---- Agent loader ----
def load_agent(pdf_path, chroma_dir, cache_path, doc_label):
    vectordb = build_vector_store(pdf_path=pdf_path, chroma_dir=chroma_dir)
    graph = build_graph(pdf_path=pdf_path, cache_path=cache_path)
    agent = build_hybrid_agent(vectordb, graph)
    st.session_state.agent = agent
    st.session_state.graph = graph
    st.session_state.agent_ready = True
    st.session_state.current_doc = doc_label


# ---- Sidebar ----
with st.sidebar:
    st.header(t("doc_mgmt"))

    # Built-in document
    st.subheader(t("builtin"))
    builtin_pdf = str(PROJECT_ROOT / "data" / "中华人民共和国个人信息保护法样例.pdf")
    builtin_chroma = str(PROJECT_ROOT / "chroma_db")
    builtin_cache = str(PROJECT_ROOT / "demo3_hybrid_agent" / "graph_cache.json")

    if st.button(t("load_builtin"), use_container_width=True):
        with st.spinner(t("building")):
            try:
                load_agent(builtin_pdf, builtin_chroma, builtin_cache, "Built-in")
            except Exception as e:
                st.error(f"{t('load_failed')}: {e}")

    st.divider()

    # Upload custom document
    st.subheader(t("upload"))
    uploaded_file = st.file_uploader(
        t("upload_prompt"), type=["pdf"], help=t("upload_help")
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

        if st.button(t("load_upload", name=doc_key), use_container_width=True):
            info = st.session_state.docs[doc_key]
            with st.spinner(t("building_for", name=doc_key)):
                try:
                    load_agent(info["path"], info["chroma"], info["cache"], doc_key)
                except Exception as e:
                    st.error(f"{t('load_failed')}: {e}")

    # Status
    st.divider()
    if st.session_state.agent_ready and st.session_state.graph:
        st.success(t("agent_ready"))
        st.metric(t("graph_nodes"), st.session_state.graph.number_of_nodes())
        st.metric(t("graph_edges"), st.session_state.graph.number_of_edges())
        st.caption(f"{t('current')}: {st.session_state.current_doc}")

    st.divider()
    lang = st.selectbox(
        t("lang_label"), ["EN", "中文"], index=0 if st.session_state.lang == "en" else 1
    )
    if lang == "EN" and st.session_state.lang != "en":
        st.session_state.lang = "en"
        st.rerun()
    elif lang == "中文" and st.session_state.lang != "cn":
        st.session_state.lang = "cn"
        st.rerun()

# ---- Main area ----
st.title(t("title"))
st.caption(t("subtitle"))

if not st.session_state.agent_ready:
    st.info(t("please_load"))
    st.subheader(t("example_label"))
    for i, ex in enumerate(t("examples"), 1):
        st.caption(f"{i}. {ex}")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    selected = st.selectbox(
        t("quick_label"),
        [""] + t("examples"),
        index=0,
        placeholder=t("quick_placeholder"),
    )
    question = selected if selected else st.chat_input(t("chat_placeholder"))

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner(t("reasoning")):
                answer = query_agent(st.session_state.agent, question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button(t("clear_chat")):
        st.session_state.messages = []
        st.rerun()