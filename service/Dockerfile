# AI Text Detection Service - CPU Version
# Minimal Docker image for running the FastAPI service

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY service/ ./service/

# Copy models directory (should be pre-trained)
# Users should mount models or copy them before building
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the service
CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
