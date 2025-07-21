import sys
from pathlib import Path

# Adiciona src/scripts ao sys.path para o import funcionar
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pandas as pd
import pytest

from gerar_dataset_treino import gerar_dataset_treino

def test_gerar_dataset_treino(tmp_path, monkeypatch):
    # Cria estrutura de diretórios fake
    parquet_dir = tmp_path / "data" / "parquet"
    (parquet_dir / "applicants").mkdir(parents=True, exist_ok=True)
    (parquet_dir / "prospects").mkdir(parents=True, exist_ok=True)

    # Dados simulados
    applicants = pd.DataFrame({
        "infos_basicas.codigo_profissional": ["A1", "B2"],
        "infos_basicas.cliente": ["XPTO", "ABC"],
        "informacoes_profissionais.nivel_profissional": ["Senior", "Junior"],
        "informacoes_profissionais.area_atuacao": ["Dados", "Infra"],
        "infos_basicas.vaga_sap": [True, False],
        "formacao_e_idiomas.nivel_ingles": ["Avançado", "Básico"],
        "formacao_e_idiomas.nivel_espanhol": ["Nenhum", "Básico"],
        "informacoes_profissionais.conhecimentos_tecnicos": ["Python", "SQL"],
        "perfil_vaga.nivel_academico": ["Superior", "Médio"]
    })
    applicants.to_parquet(parquet_dir / "applicants" / "applicants.parquet", index=False)

    prospects = pd.DataFrame({
        "prospect_id": [1],
        "codigo": ["A1"]
    })
    prospects.to_parquet(parquet_dir / "prospects" / "prospect1.parquet", index=False)

    # Monkeypatch o diretório atual para o tmp_path
    monkeypatch.chdir(tmp_path)

    # Executa o pipeline
    gerar_dataset_treino()

    # Valida se o parquet final foi gerado
    final_path = parquet_dir / "parquet_treino_unificado.parquet"
    assert final_path.exists()

    df = pd.read_parquet(final_path)
    # Valida se as principais colunas estão presentes
    for col in [
        "cliente", "nivel_profissional", "idioma_requerido", "eh_sap",
        "area_atuacao", "nivel_ingles", "nivel_espanhol", "nivel_academico",
        "conhecimentos_tecnicos", "contratado"
    ]:
        assert col in df.columns

    # Checa a coluna target
    assert set(df["contratado"].unique()).issubset({0, 1})
    # Garante que quem teve match recebeu contratado=1
    assert df.loc[df["cliente"] == "XPTO", "contratado"].iloc[0] == 1
