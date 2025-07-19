# TechChallenge5 - Projeto Datathon Bruna

Este repositório contém a solução desenvolvida para o Datathon da Decision, com foco na aplicação de técnicas de Machine Learning para otimizar o processo de recrutamento e seleção de candidatos.

---

## 📁 Estrutura do Projeto

```
projeto_datathon_brunag/
├── data/
│   └── parquet/
│       ├── applicants/         # Dados brutos de candidatos
│       ├── prospects/          # Dados de candidaturas
│       └── vagas/              # Dados de vagas
├── model/
│   ├── modelo_classificador.pkl # Modelo RandomForest treinado
│   └── label_encoder_contratado.pkl # Encoder do rótulo de contratação
├── utils/
│   ├── data_merger.py          # Função para mesclar DataFrames
│   └── paths.py                # Definição de caminhos de arquivos
├── run_train.py                # Script principal de treinamento e API
├── teste.py                    # Script auxiliar para inspecionar colunas
├── build_local.bat             # Batch script para setup e processamento
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documento de descrição do projeto
```

---

## 🚀 Pré-requisitos

* Python 3.10 ou superior
* Git
* (Windows) PowerShell

---

## ⚙️ Configuração do Ambiente

1. Clone este repositório:

   ```bash
   git clone https://github.com/bru-guimaraes/techchallenge5_bruna.git
   cd techchallenge5_bruna/projeto_datathon_brunag
   ```

2. Execute o script de setup e processamento:

   ```powershell
   .\build_local.bat
   ```

   Este script:

   * Cria e ativa um ambiente virtual
   * Instala as dependências de `requirements.txt`
   * Processa os arquivos ZIP e salva os Parquets em `data/parquet/`

---

## 🎓 Treinamento do Modelo e API

Para treinar o modelo e iniciar a API, utilize:

```bash
python run_train.py
```

O fluxo de `run_train.py` é:

1. Carregar dados de `data/parquet/`.
2. Mesclar DataFrames (`utils/data_merger.py`).
3. Gerar coluna alvo binária `contratado` a partir de `situacao_candidado`.
4. Filtrar linhas com valores ausentes em features ou alvo.
5. One-hot encoding das features categóricas.
6. Dividir em treino e teste (70/30).
7. Balancear classes com SMOTE.
8. Treinar `RandomForestClassifier`.
9. Exibir relatório de classificação e importâncias.
10. Salvar modelo em `model/modelo_classificador.pkl`.
11. Iniciar servidor FastAPI ([http://127.0.0.1:8000](http://127.0.0.1:8000)).

---

## 🛠️ API de Previsão

* **Endpoint**: `POST /predict`
* **Payload** (exemplo):

  ```json
  {
    "area_atuacao": "TI - Desenvolvimento/Programação",
    "nivel_ingles": "Intermediário",
    "nivel_espanhol": "Básico",
    "nivel_academico": "Ensino Superior Completo"
  }
  ```
* **Response**:

  ```json
  { "predicao": 1 }
  ```

  onde `1` indica provável contratação.

---

## 📊 Testes e Inspeção

* **`teste.py`**: imprime colunas dos DataFrames para auxiliar a verificar nomes exatos.

---

## 📦 Deploy & Docker

Você pode dockerizar a aplicação criando um `Dockerfile` na raiz e executando:

```bash
docker build -t datathon-bruna .
docker run -p 8000:8000 datathon-bruna
```