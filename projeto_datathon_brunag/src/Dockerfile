# Builder: gera wheels
FROM python:3.10-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Runtime: instala wheels e copia o código
FROM python:3.10-slim AS runtime
WORKDIR /app

# Definições de ambiente
ENV PATH_MODEL=model/pipeline.joblib \
    FEATURES_JSON_PATH=model/features.json \
    PREDICTION_THRESHOLD=0.5

# Executa como usuário não-root
RUN useradd --create-home appuser
USER appuser

# copia apenas as wheels e instala
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt

# copia o restante do código (sem dados pesados)
COPY --chown=appuser:appuser . .

EXPOSE 8000

# Healthcheck para o Docker
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
