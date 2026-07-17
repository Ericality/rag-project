"""
Demo 1: Vector-based RAG for Legal QA

Core approach:
- Load a legal PDF → split into chunks → embed with sentence-transformers
- Store embeddings in ChromaDB → build a LangChain Agent with retrieval tool
- Agent can autonomously decide when to search & how to answer
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ---- Configuration ----
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "./data/中华人民共和国个人信息保护法样例.pdf")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

LLM = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
    max_tokens=4096,
    temperature=0,
)


def build_vector_store():
    """Build or load the ChromaDB vector store from the target PDF."""
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DB_DIR):
        vectordb = Chroma(
            persist_directory=CHROMA_DB_DIR, embedding_function=embedding
        )
    else:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=CHROMA_DB_DIR,
        )
    return vectordb


def build_rag_agent(vectordb):
    """Create a LangChain Agent with a vector-retrieval tool."""

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    def search_law(query: str) -> str:
        """Search the Personal Information Protection Law for relevant provisions."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant provisions found."
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[{i}] {doc.page_content}")
        return "\n\n".join(parts)

    agent = create_agent(
        model=LLM,
        tools=[search_law],
        system_prompt=(
            "You are a Chinese legal Q&A assistant. Answer user questions based on "
            "the provisions of the Personal Information Protection Law. When the "
            "question involves legal provisions, call the search_law tool to retrieve "
            "relevant text, then formulate an answer based on the results. Answers "
            "should be concise, professional, and cite the relevant provisions."
        ),
    )
    return agent


def query_agent(agent, question: str) -> str:
    """Send a question to the agent and return the final answer."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    vectordb = build_vector_store()
    agent = build_rag_agent(vectordb)

    test_questions = [
        "What principles must be followed when processing personal information?",
        "Under what circumstances is consent not required?",
        "What special protections exist for minors' personal information?",
        "What rights do individuals have regarding their personal information?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"[Q] {q}")
        answer = query_agent(agent, q)
        print(f"[A] {answer}")
        print("=" * 60)