"""Composer probabilities and per-feature contributions for one piece."""
import numpy as np
import pandas as pd


def explain(bundle: dict, features: pd.DataFrame, top_k: int = 10):
    """(probabilities sorted high to low, top_k features by contribution to the predicted composer)."""
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f"missing features: {missing[:5]}")
    X = features[cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any(axis=None):
        raise ValueError(f"non-numeric features: {X.columns[X.isna().any()].tolist()[:5]}")
    pipe = bundle["pipeline"]
    values = X.to_numpy(float)
    probs = pd.Series(pipe.predict_proba(values)[0], index=pipe.classes_).sort_values(ascending=False)
    scaled = pipe.named_steps["scale"].transform(values)
    contrib = np.asarray(pipe.named_steps["clf"].booster_.predict(scaled, pred_contrib=True))
    winner = list(pipe.classes_).index(probs.index[0])
    contrib = contrib.reshape(len(pipe.classes_), len(cols) + 1)[winner, :-1]
    top = pd.DataFrame({"feature": cols, "value": values[0], "contribution": contrib})
    top = top.iloc[np.argsort(-np.abs(top.contribution.to_numpy()), kind="stable")[:top_k]]
    return probs, top.reset_index(drop=True)
