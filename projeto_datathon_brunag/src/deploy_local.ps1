# deploy_local.ps1 — rodar dentro de src/

# 0) Garante que estamos no diretório do script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

# 1) Pre-deploy: extrai JSONs e gera parquet unificado
Write-Host "[Pre-Deploy] Extraindo JSONs e gerando dataset..."
python -m utils.extrair_json_de_zip
if ($LASTEXITCODE -ne 0) { Write-Error "Falha na extração."; exit 1 }
python scripts/gerar_dataset_treino.py
if ($LASTEXITCODE -ne 0) { Write-Error "Falha na geração do dataset."; exit 1 }

# 2) Treina pipeline local
Write-Host "[Pre-Deploy] Treinando pipeline local..."
python run_train.py data/parquet/parquet_treino_unificado.parquet
if ($LASTEXITCODE -ne 0) { Write-Error "Falha no treinamento."; exit 1 }

# 3) Build da imagem Docker
$Image = "datathon-api:local"
Write-Host "[Deploy] Construindo Docker image $Image..."
docker build -t $Image .
if ($LASTEXITCODE -ne 0) { Write-Error "Falha no docker build."; exit 1 }

# 4) Para e remove container antigo
$Container = "datathon-api-local"
if (docker ps -a --format "{{.Names}}" | Select-String "^$Container$") {
    Write-Host "[Deploy] Parando/removendo container $Container..."
    docker stop $Container | Out-Null
    docker rm   $Container | Out-Null
}

# 5) Inicia novo container
Write-Host "[Deploy] Iniciando container $Container na porta 8000..."
docker run -d `
  --name $Container `
  -p 8000:8000 `
  --restart unless-stopped `
  $Image | Out-Null

Write-Host "`n[Deploy] Concluído! API disponível em http://localhost:8000/docs"
