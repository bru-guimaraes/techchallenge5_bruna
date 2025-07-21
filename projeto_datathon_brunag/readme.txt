Projeto Datathon - Machine Learning Engineering

Visão Geral

Este projeto faz parte do desafio de Machine Learning Engineering do Datathon. O objetivo é construir um pipeline completo de dados, englobando extração, transformação, modelagem preditiva, exposição via API, testes automatizados, documentação e provisionamento em nuvem.

Estrutura do Projeto

projeto_datathon_brunag/
│
├── src/
│   ├── application.py            # API FastAPI
│   ├── run_train.py              # Script de treino do modelo
│   ├── scripts/
│   │   └── gerar_dataset_treino.py # Geração do parquet de treino
│   ├── utils/
│   │   ├── extrair_json_de_zip.py   # Função de extração de JSONs
│   │   ├── carregar_dados_para_treino.py # Loader
│   │   └── paths.py                # Paths de modelo
│   ├── model/
│   │   ├── pipeline.joblib         # Modelo treinado
│   │   └── features.json           # Lista de features
│   ├── data/
│   │   └── parquet/                # Parquets intermediários
│   └── tests/
│       ├── test_api_endpoints.py
│       ├── test_data_merger.py
│       ├── test_extractor.py
│       ├── test_gerar_dataset_treino.py
│       └── test_run_train.py
├── Dockerfile
├── requirements.txt
├── README.md
└── commit.bat (automatização de testes e git)

Passos para Execução Local

1. Instalação de Dependências

pip install -r requirements.txt

2. Extração de Dados e Pré-processamento

Coloque os arquivos .zip de dados em data/raw_downloads/.

Execute o script para extração e geração dos parquets:

python -m utils.extrair_json_de_zip
python src/scripts/gerar_dataset_treino.py

3. Treino do Modelo

python run_train.py

Isso irá treinar o pipeline, gerar model/pipeline.joblib e model/features.json.

4. Rodar a API

uvicorn application:app --reload

Acesse http://127.0.0.1:8000 para interagir com a API e visualizar a documentação Swagger/OpenAPI.

5. Testes Automatizados

Na raiz do repositório:

cd src
pytest

Ou utilize o script automatizador:

commit.bat

Executa pip, dependências, testes e só faz commit se tudo passar.

Docker

Build e Run

docker build -t datathon-api .
docker run -it -p 8000:8000 datathon-api

Variáveis de Ambiente Úteis

PATH_MODEL: Caminho do pipeline salvo

FEATURES_JSON_PATH: Caminho do features.json

Testando a API

Via Swagger: http://localhost:8000/docs

Via Curl:

curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "cliente": "Empresa X",
  "nivel_profissional": "Senior",
  "idioma_requerido": "ingles",
  "eh_sap": true,
  "area_atuacao": "Dados",
  "nivel_ingles": "alto",
  "nivel_espanhol": "baixo",
  "nivel_academico": "superior",
  "conhecimentos_tecnicos": "python"
}'

Testes Automatizados

Todos os testes estão na pasta src/tests/. Eles cobrem:

Extração de parquets

Geração do dataset de treino

Pipeline de treino

Endpoints da API

Ao rodar pytest, todos são executados automaticamente.

CI/CD com GitHub Actions

Sugestão de workflow:

name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest src/tests/ --maxfail=1 --disable-warnings -q

Adapte conforme estrutura do seu projeto e caminhos.

Monitoramento e Métricas

A API expõe métricas via /metrics para Prometheus, incluindo contadores e histogramas de requisições.

Requisitos do Desafio (Checklist)



Extras e Observações

Todos os scripts e funções possuem comentários explicativos.

Automatização de commit para garantir só pushes com testes OK.

Para produção/cloud, adapte variáveis de ambiente e paths conforme necessário.

Contato

Autor: Bruna Guimarães

Repositório: [link do GitHub]

Dúvidas/sugestões: abra uma issue!

