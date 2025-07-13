#########################################
# Stage 1: Test and build on builder
#########################################
FROM python:3.10-slim AS builder

WORKDIR /app

# 1) Dependências do SO para extensões nativas (PyYAML, etc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Copia e instala dependências Python (runtime + dev)
COPY requirements.txt .
# Instala pytest para testes
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt pytest

# 3) Copia código e modelo
COPY . .

# 4) Executa testes antes de avançar
RUN pytest --maxfail=1 --disable-warnings -q

#########################################
# Stage 2: Runtime image
#########################################
FROM python:3.10-slim

WORKDIR /app

# 1) Dependências do SO para extensões nativas (PyYAML, etc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Copia e instala apenas runtime dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 3) Copia aplicação e modelo a partir do builder
COPY --from=builder /app /app

EXPOSE 8000
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
