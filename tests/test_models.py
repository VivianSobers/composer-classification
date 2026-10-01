import numpy as np
import pytest

from composer_clf.models import GRIDS, MODEL_NAMES, build


def toy_data():
    rng = np.random.default_rng(0)
    y = np.repeat(["x", "y", "z"], 20)
    X = rng.normal(size=(60, 5)) + (y == "y")[:, None] * 2 - (y == "z")[:, None] * 2
    return X, y


def test_every_model_has_a_grid():
    assert set(GRIDS) == set(MODEL_NAMES) == {"lightgbm", "softmax", "svm", "lda", "knn", "mlp"}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_grid_has_eight_configs(name):
    assert len(GRIDS[name]) == 8


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_config_fits_and_predicts(name):
    X, y = toy_data()
    for params in GRIDS[name]:
        pipe = build(name, params, seed=0).fit(X, y)
        assert set(pipe.predict(X)) <= {"x", "y", "z"}
        assert list(pipe.named_steps) == ["scale", "clf"]
