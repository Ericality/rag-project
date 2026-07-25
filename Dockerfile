FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ChromaDB & PyPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose port for Streamlit (future UI)
EXPOSE 8501

# Default command: show available demos
CMD ["python", "-c", "print('Available demos: python demo1_vector_rag/rag_qa.py | python demo2_knowledge_graph/knowledge_graph.py | python demo3_hybrid_agent/agent_graphrag.py')"]