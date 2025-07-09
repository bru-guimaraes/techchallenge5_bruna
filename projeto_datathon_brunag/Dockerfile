# 1. Base image
FROM python:3.10-slim

# 2. Diretório de trabalho
WORKDIR /app

# 3. Copia dependências
COPY requirements.txt .

# 4. Instala dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 5. Copia todo o código
COPY . .

# 6. Exponha a porta da API
EXPOSE 8000

# 7. Comando padrão: inicia o Uvicorn
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
