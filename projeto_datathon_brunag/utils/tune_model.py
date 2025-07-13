import pandas as pd
import joblib
import yaml
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    StratifiedKFold
)
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# opcional: XGBoost
from xgboost import XGBClassifier

from utils.data_merger import merge_dataframes
from utils.paths import (
    PATH_PARQUET_APPLICANTS,
    PATH_PARQUET_PROSPECTS,
    PATH_PARQUET_VAGAS,
    PATH_MODEL
)

def load_data():
    applicants = pd.read_parquet(PATH_PARQUET_APPLICANTS)
    prospects  = pd.read_parquet(PATH_PARQUET_PROSPECTS)
    vagas      = pd.read_parquet(PATH_PARQUET_VAGAS)
    df = merge_dataframes(applicants, prospects, vagas)
    df.rename(columns={
        'informacoes_profissionais.area_atuacao': 'area_atuacao',
        'formacao_e_idiomas.nivel_academico':     'nivel_academico',
        'formacao_e_idiomas.nivel_ingles':        'nivel_ingles',
        'formacao_e_idiomas.nivel_espanhol':      'nivel_espanhol'
    }, inplace=True)
    df['contratado'] = (df['situacao_candidado']=='Contratado pela Decision').astype(int)
    features = ['area_atuacao','nivel_ingles','nivel_espanhol','nivel_academico']
    df = df.dropna(subset=features+['contratado'])
    X = pd.get_dummies(df[features]).astype(float)
    y = df['contratado']
    return train_test_split(X, y,
                            test_size=0.3,
                            random_state=42,
                            stratify=y)

def run_search(estimator, param_dist, X_res, y_res, name, n_iter=30):
    """
    Se estimator for XGBClassifier, X_res/y_res já devem ser numpy arrays.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        error_score='raise'
    )
    print(f"\n>> Starting RandomizedSearch for {name} (n_iter={n_iter})")
    rs.fit(X_res, y_res)
    print(f">> Best {name} params:", rs.best_params_)
    print(f">> Best {name} CV F1-score : {rs.best_score_:.4f}")
    return rs.best_estimator_

def evaluate(model, X_test, y_test, label):
    print(f"\n>> Hold-out Evaluation for {label}:")
    y_pred = model.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=['não-contratado','contratado']
    ))
    print(f">> Hold-out AUC-ROC: "
          f"{roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")

def main():
    # 0) load data and split
    X_train, X_test, y_train, y_test = load_data()

    # 1) balance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # convert to numpy for XGB
    X_res_np, y_res_np = X_res.values, y_res.values
    X_test_np, y_test_np = X_test.values, y_test.values

    # 2) RF param grid (remove 'auto' which is invalid)
    rf_param_dist = {
        'n_estimators':      [100, 200, 300, 500],
        'max_depth':         [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['sqrt', 'log2', 0.5, None],
        'class_weight':      [None, 'balanced', 'balanced_subsample']
    }

    # 3) XGB param grid
    balance_ratio = y_res.sum()/(len(y_res)-y_res.sum())
    xgb_param_dist = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [3, 6, 9],
        'learning_rate':     [0.01, 0.05, 0.1],
        'subsample':         [0.6, 0.8, 1.0],
        'colsample_bytree':  [0.6, 0.8, 1.0],
        'scale_pos_weight':  [1, balance_ratio]
    }

    # 4) Search RF on DataFrame
    best_rf = run_search(
        RandomForestClassifier(random_state=42),
        rf_param_dist,
        X_res, y_res,
        name="RandomForest",
        n_iter=40
    )

    # 5) Search XGB on numpy arrays
    best_xgb = run_search(
        XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ),
        xgb_param_dist,
        X_res_np, y_res_np,
        name="XGBoost",
        n_iter=30
    )

    # 6) Evaluate both
    evaluate(best_rf, X_test, y_test, "RandomForest")
    evaluate(best_xgb, X_test_np, y_test_np, "XGBoost")

    # 7) Save the best by AUC-ROC
    rf_auc = roc_auc_score(
        y_test, best_rf.predict_proba(X_test)[:,1]
    )
    xgb_auc = roc_auc_score(
        y_test_np, best_xgb.predict_proba(X_test_np)[:,1]
    )
    chosen = best_rf if rf_auc >= xgb_auc else best_xgb
    joblib.dump(chosen, PATH_MODEL)
    print(f"\n✅ Saved best model "
          f"({'RF' if rf_auc>=xgb_auc else 'XGB'}) to {PATH_MODEL}")

if __name__=="__main__":
    main()
