import pandas as pd
import pytest
from utils.mesclar_dados import mesclar_dataframes

def make_dummy_dfs():
    # applicants: applicant_id, feat_a
    a = pd.DataFrame({'applicant_id': [1,2], 'feat_a': [10,20]})
    # prospects: prospect_id, applicant_id, feat_p
    p = pd.DataFrame({'prospect_id':[11,12], 'applicant_id':[1,2], 'feat_p':[0.1,0.2]})
    # vagas: prospect_id, feat_v
    v = pd.DataFrame({'prospect_id':[11,12], 'feat_v':[True, False]})
    return a, p, v

def test_merge_minimal():
    a, p, v = make_dummy_dfs()
    df = mesclar_dataframes(a, p, v)
    # verifica colunas existentes
    assert 'feat_a' in df.columns
    assert 'feat_p' in df.columns
    assert 'feat_v' in df.columns
    # tamanho: deve ter tantas linhas quanto prospects
    assert len(df) == 2
    # join correto dos IDs
    assert set(df['applicant_id']) == {1,2}
    assert set(df['prospect_id']) == {11,12}
