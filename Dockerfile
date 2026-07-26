FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (one layer per package — cached layers survive network failures)
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 langchain
RUN pip install --no-cache-dir --retries 10 --timeout 120 langchain-community
RUN pip install --no-cache-dir --retries 10 --timeout 120 langchain-text-splitters
RUN pip install --no-cache-dir --retries 10 --timeout 120 langchain-huggingface
RUN pip install --no-cache-dir --retries 10 --timeout 120 langchain-openai
RUN pip install --no-cache-dir --retries 10 --timeout 120 chromadb
RUN pip install --no-cache-dir --retries 10 --timeout 120 networkx
RUN pip install --no-cache-dir --retries 10 --timeout 120 pypdf
RUN pip install --no-cache-dir --retries 10 --timeout 120 python-dotenv
RUN pip install --no-cache-dir --retries 10 --timeout 120 streamlit
RUN pip install --no-cache-dir --retries 10 --timeout 120 sentence-transformers
RUN pip install --no-cache-dir --retries 10 --timeout 120 torchvision

# Copy project code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "demo4_web_ui/app.py", "--server.address=0.0.0.0"]
