FROM python:3.10-slim

WORKDIR /app

# 1) Dependências do SO para extensões nativas (PyYAML, etc)
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

# 3) Copia código e modelo
COPY . .

EXPOSE 8000
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
