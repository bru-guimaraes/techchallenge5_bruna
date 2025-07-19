# → builder: roda testes e gera modelo
FROM python:3.10-slim AS builder
WORKDIR /app

# 1) Copia deps e config
COPY src/requirements.txt src/config.yaml ./

# 2) Instala libs de SO, deps de build/test
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3-dev libyaml-dev \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir pytest imbalanced-learn ruamel.yaml shap

# 3) Copia todo o código
COPY src/ ./

# 4) Ajusta PYTHONPATH
ENV PYTHONPATH="/app"

# 5) Roda apenas os testes de endpoint
RUN pytest --maxfail=1 --disable-warnings -q tests/test_api_endpoints.py

# 6) Executa treino
RUN python run_train.py


# → runtime: mínimo para rodar a API
FROM python:3.10-slim AS runtime
WORKDIR /app

# 7) Copia e instala só deps de runtime
COPY src/requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir python-json-logger prometheus-client shap

# 8) Copia artefatos do builder
COPY --from=builder /app/model ./model
COPY --from=builder /app/features.json ./

# 9) Copia aplicação e utils
COPY --from=builder /app/application.py ./application.py
COPY --from=builder /app/utils ./utils

# 10) PYTHONPATH e entrypoint
ENV PYTHONPATH="/app"
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
