from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class SkillMatchTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer que calcula a similaridade de skills entre vaga e candidato
    com base em colunas de texto separadas por vírgula.
    """
    def __init__(self, job_skill_col: str, candidate_skill_col: str):
        self.job_skill_col = job_skill_col
        self.candidate_skill_col = candidate_skill_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def score(row):
            job_set = set(str(row[self.job_skill_col]).split(", "))
            cand_set = set(str(row[self.candidate_skill_col]).split(", "))
            return len(job_set & cand_set)
        return pd.DataFrame({'skill_match_score': X.apply(score, axis=1)})
