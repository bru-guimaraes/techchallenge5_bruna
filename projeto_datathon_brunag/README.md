Projeto Datathon Grupo 13

Este projeto implementa uma solução completa de Machine Learning Engineering para recrutamento e seleção com previsão de sucesso de candidatos, deploy produtivo em nuvem, monitoramento e automação.

🎯 Objetivo

Desenvolver uma solução de ML capaz de prever a adequação de candidatos a vagas de TI, disponibilizando:
Pipeline de processamento e treinamento
API para inferência de novos dados
Deploy Dockerizado em nuvem
Monitoramento e logs com AWS CloudWatch
Testes automatizados via GitHub Actions

1. Clonar o repositório

Os arquivos .zip da base de dados, originalmente fornecidos em um Google Drive pela organização do Datathon, foram baixados localmente, descompactados e já estão anexados/commitados no repositório do projeto no GitHub.
Assim, não é necessário baixar manualmente do Drive para rodar o projeto localmente – basta clonar o repositório normalmente.

git clone https://github.com/bru-guimaraes/techchallenge5_bruna.git
cd techchallenge5_bruna/src

2. Instalar Docker (Amazon Linux 2)

sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo systemctl enable docker
sudo systemctl start docker

Se receber command not found, use o script oficial do Docker:

curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker
sudo systemctl start docker

2.1. Clone o projeto do GitHub

Após instalar o Docker, faça o clone do repositório oficial deste projeto:

git clone https://github.com/bru-guimaraes/techchallenge5_bruna.git
cd techchallenge5_bruna/src

Os dados necessários (.zip já extraídos) já estão presentes no repositório.

3. Adicionar usuário ao grupo docker

sudo usermod -aG docker $USER
newgrp docker

4. Construir a imagem Docker

cd ~/techchallenge5_bruna/src
docker build -t datathon-api:latest .

5. Executar o container (Local ou EC2)

Local:
docker run -d --name datathon-api -p 8000:8000 datathon-api:latest

Na EC2 (com logs CloudWatch):
docker run -d --name datathon-api \
  -p 8000:8000 \
  --log-driver=awslogs \
  --log-opt awslogs-region=us-east-1 \
  --log-opt awslogs-group=datathon-logs \
  --log-opt awslogs-stream=api-stream \
  datathon-api:latest

Importante: Troque a porta 80:8000 por 8000:8000 se quiser acessar o Swagger (FastAPI docs) normalmente.

6. Testar a API

Abra no navegador:

http://<EC2_PUBLIC_IP>:8000/docs
Substitua <EC2_PUBLIC_IP> pelo IP público da sua instância EC2.

Ou via curl:
curl -X POST "http://<EC2_PUBLIC_IP>:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"coluna1": "valor", "coluna2": 123, ...}'

Veja exemplo real de payload em /docs.

7. Uso do arquivo .env

O projeto utiliza variáveis de ambiente para segredos, caminhos e parâmetros sensíveis.
Nunca faça commit do .env! Use o .env.example como referência.
Ajuste conforme o seu ambiente.

8. Observabilidade e Logs com AWS CloudWatch

Requisitos obrigatórios
Permissão IAM na instância EC2

A EC2 deve estar associada a uma IAM Role com permissão para publicar logs no CloudWatch.
Recomenda-se usar a policy CloudWatchAgentServerPolicy ou CloudWatchLogsFullAccess.
Grupo de logs (Log Group) no CloudWatch
O grupo de logs precisa existir antes de subir o container.
No exemplo deste projeto, o grupo utilizado é: datathon-logs.

Passo a passo para configurar
Associe a IAM Role à instância EC2
Console AWS: EC2 > Instâncias > [sua instância] > Ações > Segurança > Modificar função do IAM
Crie o Log Group no CloudWatch
Console: CloudWatch > Logs > Grupos de logs > Criar grupo de logs > datathon-logs

AWS CLI:
aws logs create-log-group --log-group-name datathon-logs --region us-east-1

9. Visualizando Logs e Monitoramento

Entre no AWS CloudWatch Console.
Abra o grupo datathon-logs.
Clique no stream api-stream para ver os logs em tempo real.
Possível criar filtros e alarmes para erros críticos.
Se o log group não existir, o container falha ao iniciar.

10. Testes automatizados e GitHub Actions

Testes locais

pytest
ou
python -m pytest

CI/CD Automático
Todos os pushes disparam testes automáticos no GitHub Actions:
Status dos testes e logs disponíveis na aba "Actions" do repositório.
O projeto só deve ser considerado "ok" para deploy se o badge estiver verde ("pass").

11. Demonstração em vídeo

https://youtu.be/aQb44KnkCxE

12. Organização do projeto

├── src/                      # Código fonte
│   ├── application.py        # API FastAPI
│   ├── run_train.py          # Script de treinamento
│   ├── scripts/              # Scripts auxiliares
│   └── utils/                # Funções utilitárias
├── requirements.txt
├── Dockerfile
├── tests/                    # Testes unitários
├── .github/workflows/        # CI/CD GitHub Actions
├── .env.example              # Exemplo de configuração sensível
├── README.md

