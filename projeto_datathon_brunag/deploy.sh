#!/usr/bin/env bash
set -euo pipefail

#
# deploy.sh — automação completa de build, push e deploy em EC2
#

# Carrega variáveis do .env (se existir)
if [[ -f .env ]]; then
  export $(grep -v '^#' .env | xargs)
fi

# Validações
: "${IMAGE_NAME:?Defina IMAGE_NAME em .env}"
: "${REGISTRY_TYPE:?Defina REGISTRY_TYPE em .env (dockerhub ou ecr)}"
if [[ "$REGISTRY_TYPE" == "ecr" ]]; then
  : "${AWS_REGION:?Defina AWS_REGION em .env para ECR}"
fi
: "${EC2_USER:?Defina EC2_USER em .env}"
: "${EC2_HOST:?Defina EC2_HOST em .env}"
: "${EC2_KEY_PATH:?Defina EC2_KEY_PATH em .env}"
: "${CONTAINER_NAME:?Defina CONTAINER_NAME em .env}"
: "${REMOTE_PORT:?Defina REMOTE_PORT em .env}"
: "${CONTAINER_PORT:?Defina CONTAINER_PORT em .env}"

LOCAL_PROJECT_DIR="."

echo "🚀 1) Construindo imagem Docker..."
docker build -t "${IMAGE_NAME}" "${LOCAL_PROJECT_DIR}"

echo "🔑 2) Fazendo push para ${REGISTRY_TYPE^^}..."
if [[ "${REGISTRY_TYPE}" == "dockerhub" ]]; then
  docker push "${IMAGE_NAME}"
else
  echo "→ Login no ECR (${AWS_REGION})"
  aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${IMAGE_NAME%%/*}"
  docker push "${IMAGE_NAME}"
fi

echo "🔗 3) Conectando no EC2 (${EC2_HOST}) para deploy..."
ssh -i "${EC2_KEY_PATH}" -o StrictHostKeyChecking=no "${EC2_USER}@${EC2_HOST}" bash -s <<EOF
  set -euo pipefail

  # 3.1) Instala Docker se não existir
  if ! command -v docker &>/dev/null; then
    echo "🐳 Instalando Docker..."
    sudo yum update -y
    sudo amazon-linux-extras install docker -y
  fi

  # 3.2) Inicia Docker
  sudo service docker start

  # 3.3) Se ECR, faz login remoto
  if [[ "${REGISTRY_TYPE}" == "ecr" ]]; then
    echo "→ Login remoto no ECR"
    aws ecr get-login-password --region "${AWS_REGION}" \
      | docker login --username AWS --password-stdin "${IMAGE_NAME%%/*}"
  fi

  # 3.4) Pull da imagem
  echo "⬇️ Pull da imagem ${IMAGE_NAME}..."
  docker pull "${IMAGE_NAME}"

  # 3.5) Para e remove container antigo, se houver
  if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
    echo "🛑 Parando container existente..."
    docker stop "${CONTAINER_NAME}"
  fi
  if docker ps -aq -f name="${CONTAINER_NAME}" | grep -q .; then
    echo "🗑️ Removendo container anterior..."
    docker rm "${CONTAINER_NAME}"
  fi

  # 3.6) Inicia novo container
  echo "⚡ Iniciando novo container..."
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -p ${REMOTE_PORT}:${CONTAINER_PORT} \
    "${IMAGE_NAME}"

  echo "🎉 Deploy concluído no EC2!"
EOF

echo "✅ Tudo pronto — API disponível em http://${EC2_HOST}:${REMOTE_PORT}/"
