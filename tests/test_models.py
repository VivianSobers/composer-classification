import numpy as np
import pytest

from composer_clf.models import GRIDS, build

IMPLEMENTED = ["lightgbm", "softmax", "svm", "lda", "knn"]


def toy_data():
    rng = np.random.default_rng(0)
    y = np.repeat(["x", "y", "z"], 20)
    X = rng.normal(size=(60, 5)) + (y == "y")[:, None] * 2 - (y == "z")[:, None] * 2
    return X, y


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_grid_has_eight_configs(name):
    assert len(GRIDS[name]) == 8


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_every_config_fits_and_predicts(name):
    X, y = toy_data()
    for params in GRIDS[name]:
        pipe = build(name, params, seed=0).fit(X, y)
        assert set(pipe.predict(X)) <= {"x", "y", "z"}
        assert list(pipe.named_steps) == ["scale", "clf"]
