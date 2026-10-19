# Serving image for the AeroOps API (delay + RUL models). Spark/training deps are NOT included.
# Build after training so models/ contains the artifacts:  make train && docker build -t aeroops-api .
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src AEROOPS_MODEL_DIR=/app/models PORT=8080
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/aeroops/__init__.py src/aeroops/__init__.py
COPY src/aeroops/serving src/aeroops/serving
COPY src/aeroops/flights/__init__.py src/aeroops/flights/schema.py src/aeroops/flights/
COPY src/aeroops/rul/__init__.py src/aeroops/rul/data.py src/aeroops/rul/features.py src/aeroops/rul/
COPY models/ models/
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn aeroops.serving.api:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS:-2}"]
