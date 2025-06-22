@echo off

echo Criando ambiente virtual...
python -m venv .venv

echo Ativando ambiente virtual...
call .venv\Scripts\activate.bat

echo Instalando dependências do requirements.txt...
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt

echo Extraindo arquivos dos .zip...
call .venv\Scripts\python.exe utils\extrair_json_de_zip.py

echo Treinando o modelo...
call .venv\Scripts\python.exe run_train.py

echo Iniciando a API...
call .venv\Scripts\uvicorn.exe application:app --reload
