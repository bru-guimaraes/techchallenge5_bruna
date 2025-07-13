FROM python:3.10-slim

WORKDIR /app

# 1) Instala dependências de SO pra compilar extensões nativas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 3) Copia o diretório 'model' antes de copiar o restante do código,
#    para aproveitar o cache do Docker caso o código mude mas o modelo não.
COPY model /app/model

# 4) Copia o restante da aplicação
COPY . .

# 5) Exponha a porta e defina o comando de inicialização
EXPOSE 8000
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
