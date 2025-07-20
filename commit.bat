@echo off
REM Vai para a pasta do próprio .bat (raiz do repo)
cd /d "%~dp0"

REM Faz stage de tudo
git add .

REM Pergunta a mensagem de commit
set /p COMMIT_MSG=Digite a mensagem do commit: 
if "%COMMIT_MSG%"=="" (
  echo Mensagem vazia. Abortando.
  exit /b 1
)

REM Commita e faz push
git commit -m "%COMMIT_MSG%" || exit /b 1
git push origin HEAD    || exit /b 1

echo Feito!
