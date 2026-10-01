import pytest

from composer_clf.evaluate import compute_metrics

Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "a", "b", "b", "b", "a"]


def test_accuracy_and_macro_hand_values():
    m = compute_metrics(Y_TRUE, Y_PRED, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["macro_recall"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert m["macro_precision"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 0) / 3)
    assert m["per_class"]["b"] == {"precision": pytest.approx(2 / 3), "recall": 1.0, "f1": pytest.approx(0.8), "support": 2}


def test_balanced_accuracy_and_confusion():
    m = compute_metrics(Y_TRUE, Y_PRED, ["a", "b", "c"])
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert m["confusion"] == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]


def test_balanced_accuracy_ignores_absent_classes():
    m = compute_metrics(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "z"])
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1.0) / 2)


def test_bootstrap_perfect_and_deterministic():
    import numpy as np

    from composer_clf.evaluate import bootstrap_ci

    y = list("aabbccaabb")
    assert bootstrap_ci(y, y) == (1.0, 1.0)
    rng = np.random.default_rng(0)
    pred = [p if rng.random() < 0.7 else "c" for p in y]
    assert bootstrap_ci(y, pred, seed=5) == bootstrap_ci(y, pred, seed=5)
    lo, hi = bootstrap_ci(y, pred, seed=5)
    point = compute_metrics(y, pred, sorted(set(y)))["accuracy"]
    assert lo <= point <= hi
