#!/bin/bash
# deploy_local.sh -- rodar dentro de src/

# Vai para a pasta do script
cd "$(dirname "$0")"

# 1) Build da imagem
IMAGE="datathon-api:local"
echo "[Deploy] Construindo Docker image $IMAGE..."
docker build -t $IMAGE .
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha no docker build."
    exit 1
fi

# 2) Para/remove container antigo
CONTAINER="datathon-api-local"
if [ "$(docker ps -a --format '{{.Names}}' | grep -w "$CONTAINER")" ]; then
    echo "[Deploy] Parando/removendo container $CONTAINER..."
    docker stop $CONTAINER >/dev/null
    docker rm   $CONTAINER >/dev/null
fi

# 3) Inicia novo container
echo "[Deploy] Iniciando container $CONTAINER na porta 8000..."
docker run -d \
  --name $CONTAINER \
  -p 8000:8000 \
  --restart unless-stopped \
  $IMAGE >/dev/null

echo
echo "[Deploy] Concluido! API disponivel em http://localhost:8000/docs"
