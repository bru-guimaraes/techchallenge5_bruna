#!/usr/bin/env python3

import os
import subprocess
import json
import yaml
import pandas as pd
import urllib.request
from pathlib import Path
import argparse


def print_status(label, ok=True):
    status = "[OK]" if ok else "[ERROR]"
    print(f"{status} {label}")


def list_directory(root: Path):
    for path in root.rglob('*'):
        print(path)


def inspect_files(data_dir: Path, model_path: Path, feats_path: Path):
    print("\n=== Inspeção de Arquivos ===")
    # Parquet files
    parquet_dir = data_dir / 'parquet'
    if parquet_dir.exists():
        print("Parquet directory contents:")
        list_directory(parquet_dir)
    else:
        print_status(f"Diretório {parquet_dir} não encontrado", ok=False)

    # Model file
    if model_path.exists():
        stat = model_path.stat()
        print(f"Modelo: {model_path} | size={stat.st_size} bytes | mtime={stat.st_mtime}")
        print_status("Modelo encontrado")
    else:
        print_status(f"Modelo não encontrado em {model_path}", ok=False)

    # Features file
    if feats_path.exists():
        print("Features.json preview (first 20 lines):")
        for i, line in enumerate(feats_path.open('r', encoding='utf-8')):
            if i < 20:
                print(line.rstrip())
            else:
                break
        print_status("Features.json encontrado")
    else:
        print_status(f"features.json não encontrado em {feats_path}", ok=False)


def inspect_parquet(unified_path: Path):
    print("\n=== Estatísticas do Parquet Unificado ===")
    try:
        df = pd.read_parquet(unified_path)
    except Exception as e:
        print_status(f"Falha ao ler {unified_path}: {e}", ok=False)
        return

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("Head:")
    print(df.head())
    print("Dtypes:")
    print(df.dtypes)
    print("Null count per column:")
    print(df.isna().sum())
    print_status("Leitura de parquet bem-sucedida")


def inspect_config(config_path: Path):
    print("\n=== Inspeção de config.yaml e PATH_MODEL ===")
    if config_path.exists():
        print("config.yaml preview (first 20 lines):")
        for i, line in enumerate(config_path.open('r', encoding='utf-8')):
            if i < 20:
                print(line.rstrip())
            else:
                break
        print_status("config.yaml encontrado")
    else:
        print_status(f"config.yaml não encontrado em {config_path}", ok=False)
    print(f"PATH_MODEL env var: {os.getenv('PATH_MODEL')}" )


def inspect_docker(container_name: str):
    print("\n=== Inspeção do Docker Container ===")
    try:
        ps = subprocess.run([
            "docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"
        ], capture_output=True, text=True, check=True)
        containers = ps.stdout.strip().splitlines()
        if container_name in containers:
            print_status(f"Container '{container_name}' está rodando")
            logs = subprocess.run(
                ["docker", "logs", "--tail", "50", container_name],
                capture_output=True, text=True, check=True
            )
            print("Últimos 50 logs:")
            print(logs.stdout)
        else:
            print_status(f"Container '{container_name}' não encontrado", ok=False)
    except subprocess.CalledProcessError as e:
        print_status(f"Erro ao inspecionar Docker: {e}", ok=False)


def test_endpoints(host: str, port: int):
    base_url = f"http://{host}:{port}"
    print("\n=== Teste de Endpoints ===")

    # Health
    try:
        resp = urllib.request.urlopen(f"{base_url}/health", timeout=5)
        print(f"/health status: {resp.getcode()} | {resp.read().decode()}")
        print_status("Health endpoint OK")
    except Exception as e:
        print_status(f"Health endpoint falhou: {e}", ok=False)

    # Predict
    payload = json.dumps({
        "cliente": "Teste",
        "nivel_profissional": "Júnior",
        "idioma_requerido": "Inglês",
        "eh_sap": False,
        "area_atuacao": "Suporte",
        "nivel_ingles": "baixo",
        "nivel_espanhol": "baixo",
        "nivel_academico": "medio",
        "conhecimentos_tecnicos": "Suporte básico"
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/predict", data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        body = resp.read().decode()
        data = json.loads(body)
        print(f"/predict status: {resp.getcode()} | resposta keys: {list(data.keys())}")
        print_status("Predict endpoint OK")
    except Exception as e:
        print_status(f"Predict endpoint falhou: {e}", ok=False)


def main():
    parser = argparse.ArgumentParser(description="Inspeciona pipeline Datathon API")
    parser.add_argument("--data-dir", default="data", help="Diretório raiz de dados")
    parser.add_argument("--model-path", default="model/modelo_classificador.joblib", help="Caminho do modelo")
    parser.add_argument("--config", default="config.yaml", help="Caminho do config.yaml")
    parser.add_argument("--container", default="datathon-api-local", help="Nome do container Docker")
    parser.add_argument("--host", default="localhost", help="Host da API")
    parser.add_argument("--port", type=int, default=8000, help="Porta da API")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    feats_path = model_path.with_name("features.json")

    inspect_files(data_dir, model_path, feats_path)
    inspect_parquet(data_dir / "parquet" / "parquet_treino_unificado.parquet")
    inspect_config(Path(args.config))
    inspect_docker(args.container)
    test_endpoints(args.host, args.port)

if __name__ == "__main__":
    main()
