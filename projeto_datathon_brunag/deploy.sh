#!/usr/bin/env bash
set -euo pipefail

# 1) Carrega .env (se existir)
if [[ -f .env ]]; then
  export $(grep -v '^#' .env | xargs)
fi

# 2) Valida variáveis
: "${IMAGE_NAME:?Defina IMAGE_NAME em .env}"
: "${REGISTRY_TYPE:?Defina REGISTRY_TYPE em .env (dockerhub ou ecr)}"
if [[ "${REGISTRY_TYPE}" == "ecr" ]]; then
  : "${AWS_REGION:?Defina AWS_REGION em .env para ECR}"
fi
: "${EC2_USER:?Defina EC2_USER em .env}"
: "${EC2_HOST:?Defina EC2_HOST em .env}"
: "${EC2_KEY_PATH:?Defina EC2_KEY_PATH em .env}"
: "${CONTAINER_NAME:?Defina CONTAINER_NAME em .env}"
: "${REMOTE_PORT:?Defina REMOTE_PORT em .env}"
: "${CONTAINER_PORT:?Defina CONTAINER_PORT em .env}"

# 3) Build & Push local
echo "🚀 Building Docker image..."
docker build -t "${IMAGE_NAME}" .

echo "🔑 Pushing to ${REGISTRY_TYPE^^}..."
if [[ "${REGISTRY_TYPE}" == "dockerhub" ]]; then
  docker push "${IMAGE_NAME}"
else
  aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${IMAGE_NAME%%/*}"
  docker push "${IMAGE_NAME}"
fi

# 4) Deploy remoto
echo "🔗 Deploying to EC2 (${EC2_HOST})..."
ssh -i "${EC2_KEY_PATH}" -o StrictHostKeyChecking=no "${EC2_USER}@${EC2_HOST}" \
  "export IMAGE_NAME=${IMAGE_NAME} REGISTRY_TYPE=${REGISTRY_TYPE} AWS_REGION=${AWS_REGION} \
   CONTAINER_NAME=${CONTAINER_NAME} REMOTE_PORT=${REMOTE_PORT} CONTAINER_PORT=${CONTAINER_PORT}; bash -s" <<'EOF'
  set -euo pipefail

  # a) AWS CLI v2
  if ! command -v aws &>/dev/null; then
    echo "🐍 Installing AWS CLI v2..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
  fi

  # b) Docker
  if ! command -v docker &>/dev/null; then
    echo "🐳 Installing Docker..."
    sudo yum update -y || sudo apt-get update -y
    # Amazon Linux / Ubuntu support
    if command -v amazon-linux-extras &>/dev/null; then
      sudo amazon-linux-extras install docker -y
    else
      sudo apt-get install docker.io -y
    fi
  fi
  sudo service docker start

  # c) Login ECR remoto (se aplicável)
  if [[ "\${REGISTRY_TYPE}" == "ecr" ]]; then
    echo "→ Logging into ECR..."
    aws ecr get-login-password --region "\${AWS_REGION}" \
      | sudo docker login --username AWS --password-stdin "\${IMAGE_NAME%%/*}"
  fi

  # d) Pull & run
  echo "⬇️ Pulling image \${IMAGE_NAME}..."
  sudo docker pull "\${IMAGE_NAME}"

  # Idempotência: stop & remove container existente
  if sudo docker ps -q -f name="\${CONTAINER_NAME}" | grep -q .; then
    echo "🛑 Stopping existing container..."
    sudo docker stop "\${CONTAINER_NAME}"
  fi
  if sudo docker ps -aq -f name="\${CONTAINER_NAME}" | grep -q .; then
    echo "🗑️ Removing existing container..."
    sudo docker rm "\${CONTAINER_NAME}"
  fi

  echo "⚡ Starting container with CloudWatch Logs integration..."
  sudo docker run -d \
    --name "\${CONTAINER_NAME}" \
    -p "\${REMOTE_PORT}:\${CONTAINER_PORT}" \
    --restart unless-stopped \
    --log-driver=awslogs \
    --log-opt awslogs-region="\${AWS_REGION}" \
    --log-opt awslogs-group="/datathon-api/logs" \
    --log-opt awslogs-stream="api" \
    "\${IMAGE_NAME}"

  echo "🎉 Deploy complete on EC2!"
EOF

echo "✅ Deployment complete — http://${EC2_HOST}:${REMOTE_PORT}/"
