FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "demo4_web_ui/app.py", "--server.address=0.0.0.0"]
