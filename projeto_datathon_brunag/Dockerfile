# → etapa de “builder” para rodar testes e gerar modelo
FROM python:3.10-slim AS builder
WORKDIR /app

# 1) Cache longo: só deps que mudam pouco
COPY requirements.txt config.yaml ./

# 2) Instala libs de SO + Python e deps de build/test
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3-dev libyaml-dev \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir pytest imbalanced-learn ruamel.yaml shap

# 3) Copia TODO o código do projeto (inclui utils/feature_engineering)
COPY . .

# 4) Ajusta PYTHONPATH para raiz e para utils, permitindo imports diretos de feature_engineering
ENV PYTHONPATH="/app:/app/utils"

# 5) Roda os testes — se algum falhar, build falha
RUN pytest --maxfail=1 --disable-warnings -q

# 6) Gera o modelo + features.json (fica em /app/model)
RUN python run_train.py


# → etapa “runtime” mínima para rodar a API
FROM python:3.10-slim AS runtime
WORKDIR /app

# 7) Instala apenas deps de runtime
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir python-json-logger prometheus-client shap

# 8) Copia a pasta completa model (contendo model/*.pkl e features.json)
COPY --from=builder /app/model ./model

# 9) Copia a aplicação e o pacote utils
COPY --from=builder /app/application.py ./application.py
COPY --from=builder /app/utils ./utils

# 10) Define PYTHONPATH e entrypoint
ENV PYTHONPATH="/app:/app/utils"
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
