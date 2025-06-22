# Projeto: Decision Match AI

Este projeto visa otimizar o processo de recrutamento usando Machine Learning.

## 🚀 Instruções

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar modelo
```bash
python run_train.py
```

Certifique-se de colocar os arquivos `applicants.zip`, `prospects.zip`, `vagas.zip` dentro da pasta `data/`.

### 3. Rodar API localmente
```bash
bash run_local.sh
```

Acesse: http://localhost:8000/docs

### 4. Docker
```bash
docker build -t decision-match .
docker run -p 8000:8000 decision-match
```

### 5. Testes
```bash
pytest tests/
```
