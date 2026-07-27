FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (chromadb has pre-built ARM64 wheels, no build-essential needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "demo4_web_ui/app.py", "--server.address=0.0.0.0"]
