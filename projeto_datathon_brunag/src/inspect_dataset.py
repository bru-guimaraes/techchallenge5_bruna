#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

def inspect_parquet(name):
    path = Path(f"data/parquet/{name}/{name}.parquet")
    if not path.exists():
        print(f"[ERROR] {path} não encontrado")
        return
    df = pd.read_parquet(path)
    print(f"{name.upper():<10}: shape={df.shape}")
    print(df.head(3), "\n")

def main():
    print("=== INSPEÇÃO DOS PARQUETS INDIVIDUAIS ===")
    for ent in ("applicants", "prospects", "vagas"):
        inspect_parquet(ent)

    print("=== CONTEÚDO DE gerar_dataset_treino.py ===")
    script_path = Path("scripts/gerar_dataset_treino.py")
    if script_path.exists():
        print(script_path.read_text())
    else:
        print("[ERROR] scripts/gerar_dataset_treino.py não encontrado")

if __name__ == "__main__":
    main()
