FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; model='BAAI/bge-reranker-v2-m3'; AutoTokenizer.from_pretrained(model); AutoModelForSequenceClassification.from_pretrained(model)"

COPY scripts /app/scripts
COPY web /app/web
COPY data/vector_cache /app/data/vector_cache

EXPOSE 8000

CMD ["uvicorn", "api:app", "--app-dir", "/app/scripts", "--host", "0.0.0.0", "--port", "8000"]
