@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM ----------------------------------------------------
REM build_local.bat — build e run da API via Docker
REM ----------------------------------------------------

echo ==============================================
echo 🚀 Stopping any running "datathon-api-local"…
echo ==============================================
docker stop datathon-api-local >nul 2>&1
docker rm datathon-api-local   >nul 2>&1

echo.
echo ==============================================
echo 🔨 Building Docker image "datathon-api:local"…
echo ==============================================
docker build --no-cache -t datathon-api:local .

if ERRORLEVEL 1 (
    echo ❌ Falha ao buildar a imagem Docker.
    exit /b 1
)

echo.
echo ==============================================
echo ▶️ Running container "datathon-api-local"…
echo ==============================================
docker run -d --name datathon-api-local -p 8000:8000 datathon-api:local

if ERRORLEVEL 1 (
    echo ❌ Falha ao iniciar o container Docker.
    exit /b 1
)

echo.
echo ==============================================
echo ✅ Container iniciado!
echo A API está disponível em: http://localhost:8000
echo Para parar o container, execute:
echo     docker stop datathon-api-local
echo ==============================================
ENDLOCAL
pause
