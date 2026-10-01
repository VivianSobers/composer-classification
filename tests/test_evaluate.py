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
