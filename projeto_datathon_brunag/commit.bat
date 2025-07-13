@echo off
setlocal enabledelayedexpansion

REM ————— Pré-check: Testes —————————

echo [1/5] Atualizando pip…
python -m pip install --upgrade pip

echo [2/5] Instalando pytest…
pip install --no-cache-dir pytest

echo [3/5] Rodando suíte de testes…
python -m pytest --maxfail=1 --disable-warnings -q
if errorlevel 1 (
  echo.
  echo ERRO: Testes falharam. Corrija os erros antes de commitar!
  exit /b 1
)

REM ————— Commit & Push ——————————

REM Pede a mensagem de commit
set /p COMMIT_MSG=Digite a mensagem do commit: 

REM Se não informou nada, sai
if "%COMMIT_MSG%"=="" (
  echo Mensagem vazia. Abortando.
  exit /b 1
)

REM Descobre o branch atual
for /f "usebackq delims=" %%b in (`git rev-parse --abbrev-ref HEAD`) do (
  set "BRANCH=%%b"
)

echo Branch atual: %BRANCH%

echo Fazendo stage de todas as alterações...
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
  echo Push falhou. Verifique se você tem permissão e conexão.
  exit /b 1
)

echo Commit e push concluídos com sucesso!
endlocal
