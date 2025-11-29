FROM python:3.11-slim

WORKDIR /app

# Install system deps (if needed later for PDFs, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY test_rag.py ./test_rag.py
COPY static ./static

# Environment
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "test_rag:app", "--host", "0.0.0.0", "--port", "8000"]


