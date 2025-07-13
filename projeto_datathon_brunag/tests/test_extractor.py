# tests/test_extractor.py

import os
import json
import zipfile
import pandas as pd
import pytest
from pathlib import Path

# importe a função de extração completa
from utils.extrair_json_de_zip import extrair_e_converter, RAW_DIR, PARQUET_DIR

@pytest.fixture(autouse=True)
def setup_dirs(tmp_path, monkeypatch):
    """
    Redireciona RAW_DIR e PARQUET_DIR para pastas temporárias.
    """
    raw = tmp_path / "raw"
    out = tmp_path / "parquet"
    raw.mkdir()
    out.mkdir()
    # monkeypatcha as constantes no módulo
    monkeypatch.setattr("utils.extrair_json_de_zip.RAW_DIR", raw)
    monkeypatch.setattr("utils.extrair_json_de_zip.PARQUET_DIR", out)
    return raw, out

def make_zip(dir_path: Path, stem: str, records: dict):
    """
    Cria um ZIP chamado <stem>.zip em dir_path, contendo um JSON "<stem>.json"
    com o conteúdo records.
    """
    zip_path = dir_path / f"{stem}.zip"
    # pasta temporária para montar o JSON
    tmp = zip_path.parent / f"tmp_{stem}"
    tmp.mkdir()
    json_file = tmp / f"{stem}.json"
    json_file.write_text(json.dumps(records), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(json_file, arcname=f"{stem}.json")
    return zip_path

def test_extrair_applicants_integration(setup_dirs):
    raw, out = setup_dirs
    # cria um ZIP de applicants com 2 registros
    sample = {
        "1": {"infos_basicas": {"codigo_profissional": "A1", "nome": "Ana"}},
        "2": {"infos_basicas": {"codigo_profissional": "B2", "nome": "Bruno"}}
    }
    make_zip(raw, "applicants", sample)

    # executa a extração completa
    extrair_e_converter()

    # verifica arquivo parquet
    pq_file = out / "applicants" / "applicants.parquet"
    assert pq_file.exists()

    df = pd.read_parquet(pq_file)
    # colunas esperadas após normalização
    assert "codigo_profissional" in df.columns
    assert "nome" in df.columns
    assert len(df) == 2

def test_extrair_prospects_integration(setup_dirs):
    raw, out = setup_dirs
    # cria ZIP de prospects: um bloco com lista de prospects
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
    assert "vaga_id" in df.columns
    # dois registros e vaga_id igual a 10
    assert len(df) == 2
    assert all(df["vaga_id"] == 10)

def test_extrair_vagas_integration(setup_dirs):
    raw, out = setup_dirs
    # cria ZIP de vagas com um bloco de informações básicas, perfil e benefícios
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
    # colunas combinadas dos três dicionários
    for col in ("titulo", "senioridade", "vale_transporte", "vaga_id"):
        assert col in df.columns
    assert len(df) == 1
    assert df.iloc[0]["vaga_id"] == 20
