import json
import zipfile
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

RAW_DIR     = Path(__file__).resolve().parent.parent / "data" / "raw_downloads"
PARQUET_DIR = Path(__file__).resolve().parent.parent / "data" / "parquet"

def extrair_applicants(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """
    Extrai e normaliza dados de applicants de um arquivo JSON dentro do ZIP,
    renomeando colunas aninhadas para um formato plano.
    """
    data = json.loads(z.read(name))
    records = list(data.values())
    df = pd.json_normalize(records)

    # Renomeia coluna de código para unificar join
    if "infos_basicas.codigo_profissional" in df.columns:
        df = df.rename(columns={"infos_basicas.codigo_profissional": "codigo_profissional"})

    # Renomeia outras colunas aninhadas em infos_basicas
    for col in list(df.columns):
        if col.startswith("infos_basicas.") and col != "infos_basicas.codigo_profissional":
            new_name = col.split(".", 1)[1]  # remove o prefixo "infos_basicas."
            df = df.rename(columns={col: new_name})

    return df

def extrair_prospects(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """
    Extrai dados de prospects, adicionando a coluna vaga_id para cada candidato.
    """
    data = json.loads(z.read(name))
    records = []
    for vaga_id, bloco in data.items():
        for cand in bloco.get("prospects", []):
            cand["vaga_id"] = int(vaga_id)
            records.append(cand)
    return pd.DataFrame(records)

def extrair_vagas(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """
    Extrai dados de vagas, achatando os subdicionários de informacoes_basicas,
    perfil_vaga e beneficios num único registro por vaga.
    """
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
    """
    Percorre todos os arquivos ZIP em RAW_DIR e, para cada um,
    aplica a função de extração correspondente (applicants, prospects ou vagas),
    salvando o resultado em Parquet em PARQUET_DIR.
    """
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
