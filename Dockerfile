FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts /app/scripts
COPY web /app/web
COPY data/vector_cache /app/data/vector_cache

EXPOSE 8000

CMD ["uvicorn", "api:app", "--app-dir", "/app/scripts", "--host", "0.0.0.0", "--port", "8000"]
