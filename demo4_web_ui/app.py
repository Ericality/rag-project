"""
Demo 4: Streamlit Web UI for Law-RAG Hybrid Agent

Run:
    streamlit run demo4_web_ui/app.py

Or via Docker:
    docker-compose up --build
"""

import sys
from pathlib import Path

# Add project root and demo dirs to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "demo1_vector_rag"))
sys.path.insert(0, str(PROJECT_ROOT / "demo2_knowledge_graph"))
sys.path.insert(0, str(PROJECT_ROOT / "demo3_hybrid_agent"))

import streamlit as st
from agent_graphrag import build_vector_store, build_graph, build_hybrid_agent, query_agent

# ---- Page config ----
st.set_page_config(page_title="Law-RAG", page_icon="⚖️", layout="wide")
st.title("⚖️ Law-RAG: Legal Intelligence System")
st.caption("Hybrid Agent — Vector Retrieval + Knowledge Graph")

# ---- Initialize agent (cached so it only loads once) ----
@st.cache_resource
def load_agent():
    """Build vector store + knowledge graph + hybrid agent. Cached across sessions."""
    with st.spinner("Building vector store..."):
        vectordb = build_vector_store()
    with st.spinner("Building knowledge graph..."):
        graph = build_graph()
    with st.spinner("Creating hybrid agent..."):
        agent = build_hybrid_agent(vectordb, graph)
    return agent, graph

agent, graph = load_agent()

# ---- Sidebar: system info ----
with st.sidebar:
    st.subheader("System Status")
    st.success("Agent ready")
    st.metric("Graph Nodes", graph.number_of_nodes())
    st.metric("Graph Edges", graph.number_of_edges())
    st.divider()
    st.markdown("**Two retrieval tools:**")
    st.markdown("- 📄 `search_law` — vector search for legal provisions")
    st.markdown("- 🔗 `search_graph_tool` — knowledge graph for concept relationships")
    st.divider()
    st.caption("Powered by LangChain Agent + DeepSeek LLM")

# ---- Chat interface ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if question := st.chat_input("Ask a legal question about China's Personal Information Protection Law..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Agent reasoning..."):
            answer = query_agent(agent, question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})