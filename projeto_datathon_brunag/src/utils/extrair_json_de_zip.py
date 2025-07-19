import os
import json
import zipfile
import pandas as pd
from pathlib import Path

# Diretórios, monkeypatched nos testes
RAW_DIR = Path(os.getenv("RAW_DIR", "raw_data"))
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", "data_parquet"))

def extrair_e_converter():
    """Extrai JSONs de cada .zip em RAW_DIR e gera um único Parquet por zip."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    for zip_path in RAW_DIR.glob("*.zip"):
        name = zip_path.stem
        out_dir = PARQUET_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Carrega e funde todos os dicionários JSON em um só
        merged = {}
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                with zf.open(member) as f:
                    data = json.load(f)
                    merged.update(data)

        # Converte para DataFrame de uma única linha
        df = pd.DataFrame([merged])

        # Grava como Parquet
        pq_file = out_dir / f"{name}.parquet"
        df.to_parquet(pq_file)
