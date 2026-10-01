"""Classifier grids. Each model gets exactly eight configurations."""
from itertools import product

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_NAMES = ("lightgbm", "softmax", "svm", "lda", "knn", "mlp")


def _grid(**axes) -> list[dict]:
    return [dict(zip(axes, values)) for values in product(*axes.values())]


GRIDS = {
    "lightgbm": _grid(n_estimators=[300, 700], reg_lambda=[1.0, 10.0], learning_rate=[0.05, 0.1]),
    "softmax": _grid(C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]),
    "svm": _grid(C=[0.5, 1.0, 2.0, 5.5, 10.0, 20.0, 50.0, 100.0]),
    "lda": _grid(shrinkage=[None, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 0.9]),
    "knn": _grid(n_neighbors=[1, 3, 5, 7, 11, 15, 21, 31]),
    "mlp": _grid(alpha=[1e-4, 1e-3, 1e-2, 1e-1], learning_rate_init=[0.01, 0.04]),
}


def _estimator(name: str, params: dict, seed: int):
    if name == "lightgbm":
        return LGBMClassifier(objective="multiclass", subsample=0.6, subsample_freq=1,
                              random_state=seed, n_jobs=4, verbose=-1, **params)
    if name == "softmax":
        return LogisticRegression(max_iter=5000, random_state=seed, **params)
    if name == "svm":
        return SVC(kernel="rbf", gamma="auto", random_state=seed, **params)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", **params)
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(150,), solver="sgd", momentum=0.9, batch_size=50,
                             max_iter=200, random_state=seed, **params)
    raise ValueError(f"unknown model: {name}")


def build(name: str, params: dict, seed: int) -> Pipeline:
    """Standardize, then the named classifier."""
    return Pipeline([("scale", StandardScaler()), ("clf", _estimator(name, params, seed))])
