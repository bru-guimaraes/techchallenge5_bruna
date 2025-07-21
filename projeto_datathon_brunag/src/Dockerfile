# Dockerfile multi-stage — roda toda a lógica de pré-deploy dentro do container

### Builder: roda extração, dataset e treino ###
FROM python:3.11-slim AS builder
WORKDIR /app

# 1) Copia requirements e instala dependências de build
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev curl \
    && pip install --upgrade pip \
    # Instala requirements para o pré-deploy (joblib, sklearn, etc)
    && pip install --no-cache-dir -r requirements.txt \
    # Instala Parquet engines para pandas
    && pip install --no-cache-dir pandas pyarrow fastparquet \
    # Gera wheels para uso posterior
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt \
    # Limpa build deps
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Copia o código-fonte
COPY . ./

# 3) Executa pré-deploy (extração, dataset, treino)
RUN python -m utils.extrair_json_de_zip && \
    python scripts/gerar_dataset_treino.py && \
    python run_train.py data/parquet/parquet_treino_unificado.parquet

### Runtime: só roda a API ###
FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PATH_MODEL=model/pipeline.joblib \
    FEATURES_JSON_PATH=model/features.json \
    PREDICTION_THRESHOLD=0.5

RUN useradd --create-home appuser

COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt

COPY --from=builder /app/data ./data
COPY --from=builder /app/model ./model
COPY --chown=appuser:appuser . ./

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
