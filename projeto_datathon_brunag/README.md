# Projeto Datathon Grupo 13

## 1. Clonar o repositório

```bash
git clone https://github.com/bru-guimaraes/techchallenge5_bruna.git
cd techchallenge5_bruna/src
```

## 2. Instalar Docker

### Para Amazon Linux 2

```bash
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo systemctl enable docker
sudo systemctl start docker
```

Se receber `command not found`, use o script oficial do Docker:

```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker
sudo systemctl start docker
```

## 3. Adicionar usuário ao grupo docker

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 4. Construir a imagem Docker

```bash
cd ~/techchallenge5_bruna/src
docker build -t datathon-api:latest .
```

## 5. Executar o container

```bash
docker run -d --name datathon-api -p 80:8000 datathon-api:latest
```

## 6. Testar a API

Abra no navegador:
```
http://<EC2_PUBLIC_IP>/
```

Substitua `<EC2_PUBLIC_IP>` pelo IP público da sua instância EC2.

---


## 7. Observabilidade e Logs com AWS CloudWatch

Este projeto pode enviar todos os logs da API rodando em Docker na EC2 diretamente para o **AWS CloudWatch Logs**, garantindo rastreabilidade e monitoramento em produção.

### Requisitos obrigatórios

1. **Permissão IAM na instância EC2**  
   A EC2 deve estar associada a uma IAM Role com permissão para publicar logs no CloudWatch.  
   - Recomenda-se usar a policy `CloudWatchAgentServerPolicy` ou `CloudWatchLogsFullAccess`.

2. **Grupo de logs (Log Group) no CloudWatch**  
   O grupo de logs precisa existir antes de subir o container.  
   - No exemplo deste projeto, o grupo utilizado é: `datathon-logs`.

### Passo a passo para configurar

#### 1. Associe a IAM Role à instância EC2

- No console AWS, navegue até **EC2 > Instâncias > [sua instância] > Ações > Segurança > Modificar função do IAM**.
- Escolha a role com as permissões acima.

#### 2. Crie o Log Group no CloudWatch

Via **Console AWS**:
- Acesse **CloudWatch > Logs > Grupos de logs > Criar grupo de logs** e nomeie como `datathon-logs`.

Via **AWS CLI**:
```bash
aws logs create-log-group --log-group-name datathon-logs --region us-east-1
