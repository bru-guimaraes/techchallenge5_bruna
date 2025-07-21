import os
import json
import zipfile
import pandas as pd
from pathlib import Path

# Define diretórios de entrada e saída (com fallback para defaults)
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw_downloads"))
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", "data/parquet"))

def extrair_e_converter():
    """
    Extrai arquivos .zip do RAW_DIR, converte os JSONs internos e salva como Parquet no PARQUET_DIR.
    Suporta dois formatos: dicts aninhados (id -> registro) e listas internas (caso prospects).
    """
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

        if not merged:
            print(f"[WARN] {zip_path} está vazio ou mal formatado, ignorando.")
            continue

        # CASO 1: dict de dicts simples (ex: applicants, vagas)
        if all(
            isinstance(v, dict) and not any(isinstance(val, list) for val in v.values())
            for v in merged.values()
        ):
            df = pd.json_normalize(list(merged.values()))
            # Exemplo: "applicants" -> "applicant_id"
            id_col = f"{name[:-1]}_id" if name.endswith("s") else f"{name}_id"
            df[id_col] = list(merged.keys())
        else:
            # CASO 2: dict de listas internas (ex: prospects)
            all_dfs = []
            for outer_id, content in merged.items():
                # procura primeiro valor que seja lista
                try:
                    list_key, lst = next((k, v) for k, v in content.items() if isinstance(v, list))
                except StopIteration:
                    print(f"[WARN] Não há lista em {outer_id} ({zip_path}), pulando.")
                    continue
                if not lst:
                    continue
                temp_df = pd.json_normalize(lst)
                id_col = f"{name[:-1]}_id" if name.endswith("s") else f"{name}_id"
                temp_df[id_col] = outer_id
                all_dfs.append(temp_df)
            if not all_dfs:
                print(f"[WARN] Nenhuma lista encontrada em {zip_path}, pulando.")
                continue
            df = pd.concat(all_dfs, ignore_index=True)

        pq_file = out_dir / f"{name}.parquet"
        df.to_parquet(pq_file, index=False)
        print(f"[Extractor] {pq_file} gerado.")

if __name__ == "__main__":
    extrair_e_converter()
