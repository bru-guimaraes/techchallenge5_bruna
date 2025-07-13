import pandas as pd
import pytest
from utils.data_merger import merge_dataframes

def make_dummy_dfs():
    # applicants: id, feat_a
    a = pd.DataFrame({'applicant_id': [1,2], 'feat_a': [10,20]})
    # prospects: id, applicant_id, feat_p
    p = pd.DataFrame({'prospect_id':[11,12], 'applicant_id':[1,2], 'feat_p':[0.1,0.2]})
    # vagas: id prospect_id, feat_v
    v = pd.DataFrame({'vagas_id':[101,102], 'prospect_id':[11,12], 'feat_v':[True, False]})
    return a, p, v

def test_merge_minimal():
    a, p, v = make_dummy_dfs()
    df = merge_dataframes(a, p, v)
    # espera colunas originais persistirem
    assert 'feat_a' in df.columns
    assert 'feat_p' in df.columns
    assert 'feat_v' in df.columns
    # tamanho: linhas = número de prospects
    assert len(df) == 2
    # join correto dos IDs
    assert set(df['applicant_id']) == {1,2}
    assert set(df['prospect_id']) == {11,12}
