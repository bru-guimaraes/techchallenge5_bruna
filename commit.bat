@echo off
setlocal enabledelayedexpansion

REM === [CHECK ROOT] Verifica se está na raiz do repositório ===
if not exist "projeto_datathon_brunag\src" (
    echo ERRO: Nao achou a pasta projeto_datathon_brunag\src. Execute este script a partir da raiz do repositorio.
    exit /b 1
)

REM === [CD SRC] Vai para a pasta src ===
cd projeto_datathon_brunag\src

REM === [STEP 1/5] Atualiza pip ===
echo [1/5] Atualizando pip...
python -m pip install --upgrade pip

REM === [STEP 2/5] Instala dependencias ===
echo [2/5] Instalando dependencias...
pip install --no-cache-dir -r requirements.txt

REM === [STEP 3/5] Rodando testes automatizados ===
echo [3/5] Rodando testes automatizados...

REM (DEBUG) Mostra caminho esperado do modelo
python -c "import os; print('Esperando modelo em:', os.path.abspath('model/pipeline.joblib'))"

REM Lista todos arquivos de teste encontrados (opcional para debug)
echo [DEBUG] Arquivos de teste encontrados:
dir /s /b test_*.py

REM Executa o pytest com parada no primeiro erro
python -m pytest --maxfail=1 --disable-warnings -q

if errorlevel 1 (
  echo.
  echo ERRO: Testes falharam. Corrija os erros antes de commitar!
  exit /b 1
)

REM === [STEP 4/5] Commit e Push ===
echo [4/5] Digite a mensagem do commit:
set /p COMMIT_MSG=

if "%COMMIT_MSG%"=="" (
  echo Mensagem vazia. Abortando.
  exit /b 1
)

for /f "usebackq delims=" %%b in (`git rev-parse --abbrev-ref HEAD`) do (
  set "BRANCH=%%b"
)

echo Branch atual: %BRANCH%
echo Fazendo stage de todas as alteracoes...
git add -A

echo Commitando...
git commit -m "%COMMIT_MSG%"

echo Puxando atualizacoes remotas...
git pull origin %BRANCH%

echo Subindo para o remoto...
git push origin %BRANCH%

echo.
echo Commit e push finalizados com sucesso.
endlocal
