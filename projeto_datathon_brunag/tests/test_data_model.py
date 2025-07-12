import pandas as pd
from utils.data_merger import merge_dataframes

def test_merge_minimal():
    a = pd.DataFrame({'id':[1],'a':[10]})
    p = pd.DataFrame({'applicant_id':[1],'p':[5]})
    v = pd.DataFrame({'prospect_id':[1],'v':[2]})
    # adapta merge_dataframes para aceitar estes nomes ou crie DataFrame realístico
    df = merge_dataframes(a,p,v)
    assert 'a' in df.columns
    assert 'p' in df.columns or 'v' in df.columns
