"""
Demo 3: Hybrid Agent — Vector Retrieval + Knowledge Graph

Core approach:
- Combine Demo 1 (vector RAG) and Demo 2 (knowledge graph) into a single Agent
- Agent receives two tools: search_law (vector) and search_graph_tool (KG)
- LLM autonomously decides which tool to invoke, in what order, based on question type
- Demonstrates Agentic GraphRAG: the Agent routes between semantic matching
  and structured relationship queries without hard-coded rules
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Add parent demo dirs to path so we can import their functions
sys.path.insert(0, str(Path(__file__).parent.parent / "demo1_vector_rag"))
sys.path.insert(0, str(Path(__file__).parent.parent / "demo2_knowledge_graph"))

from knowledge_graph import (
    extract_triples_from_chunk,
    build_knowledge_graph,
    search_graph,
    format_graph_results,
)

# ---- Configuration ----
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "./data/中华人民共和国个人信息保护法样例.pdf")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GRAPH_CACHE = Path("demo3_hybrid_agent/graph_cache.json")

LLM = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
    max_tokens=4096,
    temperature=0,
)

# ---- Vector Store (from Demo 1) ----


def build_vector_store(pdf_path=None, chroma_dir=None):
    """Build or load the ChromaDB vector store.

    Args:
        pdf_path: Path to the PDF file. Defaults to PDF_PATH env var.
        chroma_dir: Directory to persist ChromaDB. Defaults to CHROMA_DB_DIR env var.
    """
    pdf_path = pdf_path or PDF_PATH
    chroma_dir = chroma_dir or CHROMA_DB_DIR

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=30, separators=["\n", "。", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(chroma_dir):
        vectordb = Chroma(
            persist_directory=chroma_dir, embedding_function=embedding
        )
    else:
        vectordb = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=chroma_dir
        )
    return vectordb


# ---- Knowledge Graph (from Demo 2, with caching) ----


def build_graph(pdf_path=None, cache_path=None):
    """Build the knowledge graph, caching triples to disk to avoid re-extraction.

    Args:
        pdf_path: Path to the PDF file. Defaults to PDF_PATH env var.
        cache_path: Path to the graph cache JSON. Defaults to GRAPH_CACHE.
    """
    pdf_path = pdf_path or PDF_PATH
    cache = Path(cache_path) if cache_path else GRAPH_CACHE

    if cache.exists():
        with open(cache) as f:
            triples = [tuple(t) for t in json.load(f)]
        return build_knowledge_graph(triples)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    all_triples = []
    for chunk in chunks[:10]:
        text = chunk.page_content.strip()
        if len(text) < 30:
            continue
        all_triples.extend(extract_triples_from_chunk(text))

    unique = [list(t) for t in set(all_triples)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(unique, f, ensure_ascii=False)

    return build_knowledge_graph(unique)


# ---- Agent Tools ----


def build_hybrid_agent(vectordb, graph):
    """Create an Agent with two tools: vector retrieval + graph query."""

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    def search_law(query: str) -> str:
        """
        Search the Personal Information Protection Law using vector retrieval.
        Use this tool when the user asks about what the law SAYS —
        e.g., definitions, specific provisions, conditions, obligations.
        Returns relevant text chunks ranked by semantic similarity.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant provisions found."
        parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
        return "\n\n".join(parts)

    def search_graph_tool(concept: str) -> str:
        """
        Query the legal knowledge graph for concept relationships.
        Use this tool when the user asks about how concepts RELATE —
        e.g., hierarchies (subcategories, parent concepts), exceptions,
        obligations ("what does X require?"), and prohibitions ("what is forbidden?").
        Provide a single concept name as the query (e.g., "consent", "personal information").
        Returns structured parent/child relations and indirect links.
        """
        results = search_graph(graph, concept, direction="both", max_depth=3)
        return format_graph_results(results)

    agent = create_agent(
        model=LLM,
        tools=[search_law, search_graph_tool],
        system_prompt=(
            "You are a Chinese legal Q&A assistant. You have TWO tools at your disposal:\n"
            "1. search_law — for finding WHAT the law says (definitions, provisions, conditions).\n"
            "2. search_graph_tool — for finding HOW concepts RELATE (hierarchies, exceptions, obligations).\n\n"
            "STRATEGY:\n"
            "- For questions about specific provisions or definitions, use search_law first.\n"
            "- For questions about concept relationships (subcategories, exceptions, what X requires), "
            "use search_graph_tool first.\n"
            "- If the first tool's results are insufficient, try the other tool.\n"
            "- If both tools return useful but different information, synthesize them.\n"
            "- If neither tool provides adequate information, honestly state that the answer "
            "cannot be found in the available legal text.\n\n"
            "RESPONSE FORMAT:\n"
            "After your answer, include a 'References' section that quotes the relevant "
            "original text passages returned by the tools. Use the format:\n"
            "---\n"
            "**References:**\n"
            "[1] <original text excerpt>\n"
            "[2] <original text excerpt>\n\n"
            "Answer concisely and professionally."
        ),
    )
    return agent


def query_agent(agent, question: str) -> str:
    """Send a question to the agent and return the final answer."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


# ---- Demo ----
if __name__ == "__main__":
    print("=" * 60)
    print("Demo 3: Hybrid Agent — Vector + Knowledge Graph")
    print("=" * 60)

    print("\n[1/3] Building vector store...")
    vectordb = build_vector_store()

    print("[2/3] Building knowledge graph...")
    graph = build_graph()
    print(f"      Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("[3/3] Creating hybrid agent...")
    agent = build_hybrid_agent(vectordb, graph)

    test_questions = [
        # Vector-leaning: factual provision lookup
        "What principles must be followed when processing personal information?",
        # Graph-leaning: hierarchy / exception query
        "What are the situations where consent is NOT required?",
        # Hybrid: requires both definition (vector) + relationship (graph)
        "How is consent defined, and what are its legal requirements?",
        # Graph-leaning: subcategory query
        "What obligations does a personal information processor have?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"[Q] {q}")
        answer = query_agent(agent, q)
        print(f"[A] {answer}")
        print("=" * 60)

    print("\nDone.")