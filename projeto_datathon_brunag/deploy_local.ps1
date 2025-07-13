# deploy_local.ps1

# 1) Nome da imagem e do container
$IMAGE_NAME     = "datathon-api:local"
$CONTAINER_NAME = "datathon-api-local"

# 2) Portas
$HOST_PORT      = 8000
$CONTAINER_PORT = 8000

Write-Host "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

# 3) Para e remove container antigo (se existir)
Write-Host "Verificando container existente: $CONTAINER_NAME"
if (docker ps -a --format "{{.Names}}" | Select-String "^$CONTAINER_NAME$") {
    Write-Host "Parando container existente..."
    try { docker stop $CONTAINER_NAME 2>$null } catch {}

    Write-Host "Removendo container existente..."
    try { docker rm $CONTAINER_NAME 2>$null } catch {}
}

# 4) Sobe o novo container
Write-Host "Iniciando container $CONTAINER_NAME na porta $HOST_PORT"
docker run -d `
    --name $CONTAINER_NAME `
    -p "$HOST_PORT`:$CONTAINER_PORT" `
    --restart unless-stopped `
    $IMAGE_NAME | Out-Null

Write-Host "Container '$CONTAINER_NAME' rodando em http://localhost:$HOST_PORT/docs"
