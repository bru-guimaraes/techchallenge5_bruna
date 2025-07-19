"""
utils/ajustar_modelo.py

Busca randômica de hiperparâmetros para RandomForest e XGBoost,
comparando por AUC e salvando o melhor modelo.
"""

import pandas as pd
import joblib
import yaml
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from .caminhos import CAMINHO_MODELO

def main() -> None:
    """Executa o tuning e salva o melhor modelo."""
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    df = pd.read_parquet(cfg["paths"]["parquet_treino_unificado"])
    X = df.drop(cfg["features"]["target_column"], axis=1)
    y = df[cfg["features"]["target_column"]]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["train"]["random_state"],
        stratify=y
    )
    smote = SMOTE(random_state=cfg["train"]["random_state"])
    X_res, y_res = smote.fit_resample(X_tr, y_tr)
    print("🛠️ Dados balanceados:", Counter(y_res))

    rf = RandomForestClassifier(random_state=42)
    busca_rf = RandomizedSearchCV(
        rf,
        cfg["tuning"]["rf_param_dist"],
        n_iter=cfg["tuning"]["n_iter"],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        random_state=42
    )
    busca_rf.fit(X_res, y_res)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    busca_xgb = RandomizedSearchCV(
        xgb,
        cfg["tuning"]["xgb_param_dist"],
        n_iter=cfg["tuning"]["n_iter"],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        random_state=42
    )
    busca_xgb.fit(X_res, y_res)

    auc_rf  = roc_auc_score(y_te, busca_rf.predict_proba(X_te)[:,1])
    auc_xgb = roc_auc_score(y_te, busca_xgb.predict_proba(X_te)[:,1])
    melhor = busca_rf.best_estimator_ if auc_rf >= auc_xgb else busca_xgb.best_estimator_

    joblib.dump(melhor, CAMINHO_MODELO)
    escolhido = "RandomForest" if auc_rf >= auc_xgb else "XGBoost"
    print(f"✅ Melhor modelo ({escolhido}) salvo em {CAMINHO_MODELO}")
