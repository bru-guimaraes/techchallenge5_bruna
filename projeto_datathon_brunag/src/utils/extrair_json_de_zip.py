import os
import json
import zipfile
import pandas as pd
from pathlib import Path

# ← Ajuste para onde os seus ZIPs realmente estão:
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw_downloads"))
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", "data/parquet"))

def extrair_e_converter():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    for zip_path in RAW_DIR.glob("*.zip"):
        name = zip_path.stem
        out_dir = PARQUET_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        merged = {}
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                with zf.open(member) as f:
                    data = json.load(f)
                    merged.update(data)

        # caso “mapa de id → registro”
        if all(
            isinstance(v, dict) and not any(isinstance(val, list) for val in v.values())
            for v in merged.values()
        ):
            df = pd.json_normalize(list(merged.values()))
            df[f"{name[:-1]}_id"] = list(merged.keys())
        else:
            # caso prospects com lista interna
            outer_id, content = next(iter(merged.items()))
            list_key, lst = next((k, v) for k, v in content.items() if isinstance(v, list))
            df = pd.json_normalize(lst)
            df[f"{name[:-1]}_id"] = outer_id

        pq_file = out_dir / f"{name}.parquet"
        df.to_parquet(pq_file, index=False)
        print(f"[Extractor] {pq_file} gerado.")

if __name__ == "__main__":
    extrair_e_converter()
