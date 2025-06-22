import json
import zipfile
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

RAW_DIR     = Path(__file__).resolve().parent.parent / "data" / "raw_downloads"
PARQUET_DIR = Path(__file__).resolve().parent.parent / "data" / "parquet"

def extrair_applicants(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    data = json.loads(z.read(name))
    records = list(data.values())
    df = pd.json_normalize(records)
    # renomeia a coluna aninhada para unificar o join
    if "infos_basicas.codigo_profissional" in df.columns:
        df = df.rename(
            columns={"infos_basicas.codigo_profissional": "codigo_profissional"}
        )
    return df

def extrair_prospects(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    data = json.loads(z.read(name))
    records = []
    for vaga_id, bloco in data.items():
        for cand in bloco.get("prospects", []):
            cand["vaga_id"]    = int(vaga_id)
            records.append(cand)
    return pd.DataFrame(records)

def extrair_vagas(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    data = json.loads(z.read(name))
    records = []
    for vaga_id, bloco in data.items():
        flat = {"vaga_id": int(vaga_id)}
        flat.update(bloco.get("informacoes_basicas", {}))
        flat.update(bloco.get("perfil_vaga", {}))
        flat.update(bloco.get("beneficios", {}))
        records.append(flat)
    return pd.DataFrame(records)

def extrair_e_converter():
    for zip_path in RAW_DIR.glob("*.zip"):
        logging.info(f"Processando: {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as z:
            stem = zip_path.stem  # 'applicants', 'prospects' ou 'vagas'
            for name in z.namelist():
                if not name.lower().endswith(".json"):
                    continue

                if stem == "applicants":
                    df = extrair_applicants(z, name)
                elif stem == "prospects":
                    df = extrair_prospects(z, name)
                elif stem == "vagas":
                    df = extrair_vagas(z, name)
                else:
                    logging.warning(f"Zip desconhecido: {stem}, pulando.")
                    continue

                out_dir = PARQUET_DIR / stem
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{stem}.parquet"
                df.to_parquet(out_file, index=False)
                logging.info(f"Salvo em: {out_file}")

if __name__ == "__main__":
    extrair_e_converter()
