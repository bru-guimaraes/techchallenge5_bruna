FROM python:3.10-slim

# 1) Define o diretório de trabalho
WORKDIR /app

# 2) Copia e instala dependências
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 3) Copia todo o código
COPY . .

# 4) Gera o modelo e o features.json durante o build
RUN python run_train.py

# 5) Expõe a porta da API
EXPOSE 8000

# 6) Comando padrão para iniciar
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
