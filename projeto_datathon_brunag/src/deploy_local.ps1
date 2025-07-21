# deploy_local.ps1 -- rodar dentro de src/

# Vai para a pasta do script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

# 1) Build da imagem
$Image = "datathon-api:local"
Write-Host "[Deploy] Construindo Docker image $Image..."
docker build -t $Image .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha no docker build."
    exit 1
}

# 2) Para/remove container antigo
$Container = "datathon-api-local"
if (docker ps -a --format "{{.Names}}" | Select-String "^$Container$") {
    Write-Host "[Deploy] Parando/removendo container $Container..."
    docker stop $Container | Out-Null
    docker rm   $Container | Out-Null
}

# 3) Inicia novo container
Write-Host "[Deploy] Iniciando container $Container na porta 8000..."
docker run -d `
  --name $Container `
  -p 8000:8000 `
  --restart unless-stopped `
  $Image | Out-Null

Write-Host "`n[Deploy] Concluido! API disponivel em http://localhost:8000/docs"
