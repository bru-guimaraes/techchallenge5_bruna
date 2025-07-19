import pandas as pd
import pytest
from utils.mesclar_dados import mesclar_dataframes

def make_dummy_dfs():
    # applicants: coluna 'id' e feature 'a'
    a = pd.DataFrame({'id': [1], 'a': [10]})
    # prospects: inclui 'prospect_id', 'applicant_id' e feature 'p'
    p = pd.DataFrame({'prospect_id': [1], 'applicant_id': [1], 'p': [5]})
    # vagas: mapeia 'prospect_id' e feature 'v'
    v = pd.DataFrame({'prospect_id': [1], 'v': [2]})
    return a, p, v

def test_merge_minimal():
    # 1) Cria DataFrames de exemplo
    a, p, v = make_dummy_dfs()
    # 2) Executa o merge
    df = mesclar_dataframes(a, p, v)

    # 3) As colunas de feature devem estar presentes
    assert 'a' in df.columns    # do applicants
    assert 'p' in df.columns    # do prospects
    assert 'v' in df.columns    # do vagas

    # 4) Deve produzir uma linha por prospect
    assert len(df) == 1

    # 5) applicant_id e prospect_id devem estar no resultado
    assert 'applicant_id' in df.columns
    assert 'prospect_id' in df.columns

    # 6) Verifica valores de ID e features
    row = df.iloc[0]
    assert row['applicant_id'] == 1
    assert row['prospect_id'] == 1
    assert row['a'] == 10
    assert row['p'] == 5
    assert row['v'] == 2
