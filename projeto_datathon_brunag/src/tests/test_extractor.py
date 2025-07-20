import os
import json
import zipfile
import pandas as pd
import pytest
from pathlib import Path

from utils.extrair_json_de_zip import extrair_e_converter, RAW_DIR, PARQUET_DIR

@pytest.fixture(autouse=True)
def setup_dirs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    out = tmp_path / "parquet"
    raw.mkdir()
    out.mkdir()
    monkeypatch.setattr("utils.extrair_json_de_zip.RAW_DIR", raw)
    monkeypatch.setattr("utils.extrair_json_de_zip.PARQUET_DIR", out)
    return raw, out

def make_zip(dir_path: Path, stem: str, records: dict):
    zip_path = dir_path / f"{stem}.zip"
    tmp = zip_path.parent / f"tmp_{stem}"
    tmp.mkdir()
    json_file = tmp / f"{stem}.json"
    json_file.write_text(json.dumps(records), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(json_file, arcname=f"{stem}.json")
    return zip_path

def test_extrair_applicants_integration(setup_dirs):
    raw, out = setup_dirs
    sample = {
        "1": {"infos_basicas": {"codigo_profissional": "A1", "nome": "Ana"}},
        "2": {"infos_basicas": {"codigo_profissional": "B2", "nome": "Bruno"}}
    }
    make_zip(raw, "applicants", sample)
    extrair_e_converter()
    pq_file = out / "applicants" / "applicants.parquet"
    assert pq_file.exists()
    df = pd.read_parquet(pq_file)
    assert "infos_basicas.codigo_profissional" in df.columns
    assert "infos_basicas.nome" in df.columns
    assert len(df) == 2

def test_extrair_prospects_integration(setup_dirs):
    raw, out = setup_dirs
    sample = {
        "10": {"prospects": [
            {"nome": "Carlos", "idade": 30},
            {"nome": "Diana", "idade": 28}
        ]}
    }
    make_zip(raw, "prospects", sample)
    extrair_e_converter()
    pq_file = out / "prospects" / "prospects.parquet"
    assert pq_file.exists()
    df = pd.read_parquet(pq_file)
    assert "nome" in df.columns
    assert "idade" in df.columns
    assert "prospect_id" in df.columns
    assert len(df) == 2
    assert all(df["prospect_id"].astype(int) == 10)

def test_extrair_vagas_integration(setup_dirs):
    raw, out = setup_dirs
    sample = {
        "20": {
            "informacoes_basicas": {"titulo": "Dev"},
            "perfil_vaga": {"senioridade": "pleno"},
            "beneficios": {"vale_transporte": True}
        }
    }
    make_zip(raw, "vagas", sample)
    extrair_e_converter()
    pq_file = out / "vagas" / "vagas.parquet"
    assert pq_file.exists()
    df = pd.read_parquet(pq_file)
    for col in ("informacoes_basicas.titulo", "perfil_vaga.senioridade", "beneficios.vale_transporte", "vaga_id"):
        assert col in df.columns
    assert len(df) == 1
    assert int(df.iloc[0]["vaga_id"]) == 20
