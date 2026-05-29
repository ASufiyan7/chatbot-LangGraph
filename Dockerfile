FROM python:3.11-slim

LABEL maintainer="NEXUS"
LABEL description="4-agent autonomous coding system"
LABEL version="2.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python modules
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all code files
COPY . .

RUN mkdir -p /app/chroma_db /app/memory

# Expose Render's standard environment port variable
EXPOSE 10000

# Start the Streamlit app using the dynamic environment variable string
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.enableXsrfProtection=false --server.enableCORS=false"]
