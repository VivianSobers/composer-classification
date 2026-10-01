import numpy as np
import pandas as pd
import pytest

from composer_clf.models import build
from composer_clf.predict import explain


@pytest.fixture
def bundle():
    rng = np.random.default_rng(0)
    y = np.repeat(["bach", "chopin", "mozart"], 30)
    X = rng.normal(size=(90, 12)) + (y == "chopin")[:, None] * 3
    cols = [f"F_{i}" for i in range(12)]
    pipe = build("lightgbm", {"n_estimators": 30, "reg_lambda": 1.0, "learning_rate": 0.1}, seed=0).fit(X, y)
    return {"pipeline": pipe, "feature_columns": cols, "labels": sorted(set(y))}


def test_explain_probabilities_and_contributions(bundle):
    row = pd.DataFrame([dict(zip(bundle["feature_columns"], [3.0] * 12))])
    probs, top = explain(bundle, row)
    assert probs.sum() == pytest.approx(1.0)
    assert probs.index[0] == "chopin"
    assert list(top.columns) == ["feature", "value", "contribution"]
    assert len(top) == 10
    assert top.contribution.abs().is_monotonic_decreasing


def test_explain_rejects_missing_features(bundle):
    with pytest.raises(ValueError, match="F_11"):
        explain(bundle, pd.DataFrame([{f"F_{i}": 0.0 for i in range(11)}]))


def test_explain_rejects_non_numeric_features(bundle):
    row = {f"F_{i}": 0.0 for i in range(12)}
    row["F_3"] = "n/a"
    with pytest.raises(ValueError, match="F_3"):
        explain(bundle, pd.DataFrame([row]))
