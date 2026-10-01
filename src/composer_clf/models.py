"""Classifier grids. Each model gets exactly eight configurations."""
from itertools import product

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _grid(**axes) -> list[dict]:
    return [dict(zip(axes, values)) for values in product(*axes.values())]


GRIDS = {
    "lightgbm": _grid(n_estimators=[300, 700], reg_lambda=[1.0, 10.0], learning_rate=[0.05, 0.1]),
    "softmax": _grid(C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]),
}


def _estimator(name: str, params: dict, seed: int):
    if name == "lightgbm":
        return LGBMClassifier(objective="multiclass", subsample=0.6, subsample_freq=1,
                              random_state=seed, n_jobs=4, verbose=-1, **params)
    if name == "softmax":
        return LogisticRegression(max_iter=5000, random_state=seed, **params)
    raise ValueError(f"unknown model: {name}")


def build(name: str, params: dict, seed: int) -> Pipeline:
    """Standardize, then the named classifier."""
    return Pipeline([("scale", StandardScaler()), ("clf", _estimator(name, params, seed))])
