FROM python:3.10-slim

# Evita buffer e bytecode
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1) Instala dependências
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2) Copia todo o código, o config e os dados brutos (Parquet ou ZIPs)
#    Certifique-se de que data/raw_downloads e data/parquet NÃO estejam no .dockerignore
COPY . .

# 3) Expõe porta da API
EXPOSE 8000

# 4) Comando padrão: extrai, treina usando config.yaml e inicia a API
CMD python utils/extrair_json_de_zip.py && \
    python run_train.py && \
    uvicorn application:app --host 0.0.0.0 --port 8000
