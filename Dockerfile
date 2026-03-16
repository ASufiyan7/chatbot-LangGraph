
FROM python:3.11-slim

# Metadata
LABEL maintainer="NEXUS"
LABEL description="4-agent autonomous coding system — Groq + Ollama + ChromaDB"
LABEL version="2.0.0"


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

#  Working directory 
WORKDIR /app

#  Python dependencies 
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

#  Application code 
COPY . .

RUN mkdir -p /app/chroma_db /app/memory

EXPOSE 8000

#  Health check 
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

#  Start the server 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]