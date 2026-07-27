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

EXPOSE 8501

# Launch Streamlit web UI
CMD ["streamlit", "run", "demo4_web_ui/app.py", "--server.address=0.0.0.0"]
