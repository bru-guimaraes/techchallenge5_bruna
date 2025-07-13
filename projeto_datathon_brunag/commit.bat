@echo off
setlocal enabledelayedexpansion

REM ----- Pre-check: Testes -----

echo [1/4] Atualizando pip...
python -m pip install --upgrade pip

echo [2/4] Instalando dependencias...
pip install --no-cache-dir -r requirements.txt

echo [3/4] Rodando suite de testes...
python -m pytest --maxfail=1 --disable-warnings -q
if errorlevel 1 (
  echo.
  echo ERRO: Testes falharam. Corrija os erros antes de commitar!
  exit /b 1
)

REM ----- Commit & Push -----

set /p COMMIT_MSG=Digite a mensagem do commit: 

if "%COMMIT_MSG%"=="" (
  echo Mensagem vazia. Abortando.
  exit /b 1
)

for /f "usebackq delims=" %%b in (`git rev-parse --abbrev-ref HEAD`) do (
  set "BRANCH=%%b"
)

echo Branch atual: %BRANCH%
echo Fazendo stage de todas as alteracoes...
git add .

echo Commitando com a mensagem:
echo   "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
  echo Algo deu errado no commit. Verifique as mensagens acima.
  exit /b 1
)

echo Enviando para origin/%BRANCH%...
git push origin %BRANCH%
if errorlevel 1 (
  echo Push falhou. Verifique se voce tem permissao e conexao.
  exit /b 1
)

echo Commit e push concluidos com sucesso!
endlocal
