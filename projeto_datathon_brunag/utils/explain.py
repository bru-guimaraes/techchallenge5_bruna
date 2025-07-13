import shap
from .predict import preprocess, _model
from .paths import load_feature_names

_feature_names = load_feature_names()
_explainer = shap.TreeExplainer(_model)

def explain_single(data: dict, threshold: float) -> dict:
    df = preprocess(data)
    probs = _model.predict_proba(df)[:, 1]
    p = float(probs[0])
    pred = int(p >= threshold)
    shap_vals = _explainer.shap_values(df)[1][0]
    return {
        "prediction": pred,
        "probability": p,
        "explanation": dict(zip(_feature_names, map(float, shap_vals)))
    }

def global_importance() -> dict:
    try:
        imps = _model.feature_importances_
        return dict(zip(_feature_names, imps.tolist()))
    except AttributeError:
        sample = preprocess({f:0 for f in _feature_names})
        shap_vals = _explainer.shap_values(sample)[1][0]
        return dict(zip(_feature_names, map(abs, shap_vals)))
