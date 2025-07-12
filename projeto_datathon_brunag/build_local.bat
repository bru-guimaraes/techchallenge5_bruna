@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM ----------------------------------------------------
REM build_local.bat — build, run and test API via Docker
REM ----------------------------------------------------

echo ====================================================
echo Building Docker image "datathon-api:local"...
echo ====================================================
docker build --no-cache -t datathon-api:local .

if %ERRORLEVEL% neq 0 (
    echo Docker build failed.
    pause
    exit /b 1
)

echo.
echo ====================================================
echo Stopping old container "datathon-api-local"...
echo ====================================================
docker stop datathon-api-local >nul 2>&1
docker rm   datathon-api-local >nul 2>&1

echo.
echo ====================================================
echo Running container "datathon-api-local"...
echo ====================================================
docker run -d --name datathon-api-local -p 8000:8000 datathon-api:local

if %ERRORLEVEL% neq 0 (
    echo Docker run failed.
    pause
    exit /b 1
)

echo.
echo ====================================================
echo Waiting for API to be healthy...
echo ====================================================
:WAIT_LOOP
curl --silent --fail http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% neq 0 (
    timeout /t 5 >nul
    goto WAIT_LOOP
)
echo API is healthy!

echo.
echo ====================================================
echo Testing endpoint /predict...
echo ====================================================
curl --silent -X POST http://localhost:8000/predict -H "Content-Type: application/json" ^
    -d "{\"area_atuacao\":\"Vendas\",\"nivel_ingles\":\"medio\",\"nivel_espanhol\":\"baixo\",\"nivel_academico\":\"superior\"}"
echo.

echo ====================================================
echo Showing last 20 logs of the container...
echo ====================================================
docker logs --tail 20 datathon-api-local

echo.
echo ====================================================
echo Done! API is at http://localhost:8000
echo To stop: docker stop datathon-api-local
echo ====================================================
ENDLOCAL
pause
