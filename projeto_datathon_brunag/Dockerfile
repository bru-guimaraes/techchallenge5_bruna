FROM python:3.10-slim

# Evita buffer do Python nos logs e bytecode desnecessário
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 1) Define o diretório de trabalho
WORKDIR /app

# 2) Copia e instala dependências
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3) Copia todo o código
COPY . .

# 4) Gera o modelo e o features.json durante o build
RUN python run_train.py

# 5) Expõe a porta da API
EXPOSE 8000

# 6) (Opcional) Healthcheck — requer curl instalado:
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
#    CMD curl -f http://localhost:8000/health || exit 1

# 7) Comando padrão para iniciar
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
