# Builder
FROM python:3.10-slim AS builder
WORKDIR /app

COPY . .

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3-dev libyaml-dev \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Extrai JSONs, gera parquet e treina o pipeline
RUN python -m utils.extrair_json_de_zip \
 && python scripts/gerar_dataset_treino.py \
 && python run_train.py data/parquet/parquet_treino_unificado.parquet

# Runtime
FROM python:3.10-slim AS runtime
WORKDIR /app

COPY --from=builder /app /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
