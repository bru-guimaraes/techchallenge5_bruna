@echo off
setlocal enabledelayedexpansion

REM ————— Pré-check: Testes —————————

echo [1/5] Atualizando o pip...
python -m pip install --upgrade pip

echo [2/5] Instalando pytest...
pip install --no-cache-dir pytest

echo [3/5] Executando a suíte de testes...
python -m pytest --maxfail=1 --disable-warnings -q
if errorlevel 1 (
  echo.
  echo ERRO: Os testes falharam. Corrija os erros antes de commitar!
  exit /b 1
)

REM ————— Commit & Push ——————————

REM Solicita a mensagem de commit
set /p COMMIT_MSG=Digite a mensagem do commit: 

REM Se não informou nada, aborta
if "%COMMIT_MSG%"=="" (
  echo Mensagem vazia. Abortando.
  exit /b 1
)

REM Descobre o branch atual
for /f "usebackq delims=" %%b in (`git rev-parse --abbrev-ref HEAD`) do (
  set "BRANCH=%%b"
)

echo Branch atual: %BRANCH%

echo [4/5] Fazendo stage de todas as alterações...
git add .

echo [5/5] Commitando com a mensagem:
echo   "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
  echo ERRO: Falha ao commitar. Verifique as mensagens acima.
  exit /b 1
)

echo Enviando para origin/%BRANCH%...
git push origin %BRANCH%
if errorlevel 1 (
  echo ERRO: Falha no push. Verifique sua permissão e conexão.
  exit /b 1
)

echo Commit e push concluídos com sucesso!
endlocal
