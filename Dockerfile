FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    fastapi==0.108.0 \
    uvicorn[standard]==0.25.0 \
    pydantic==2.5.3

WORKDIR /app

COPY serve_embedding.py /app/

EXPOSE 8080

ENV PORT=8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "serve_embedding.py"]
