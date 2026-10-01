import numpy as np
import torch

from composer_clf.cnn.model import ConvNet


def test_forward_shape_and_flat_size():
    net = ConvNet(6)
    assert net.FLAT == 6144
    assert net(torch.zeros(2, 1, 64, 1536)).shape == (2, 6)


def test_predict_uses_eval_mode_and_batches():
    from composer_clf.cnn.train import predict

    torch.manual_seed(0)
    net = ConvNet(3)
    X = np.random.default_rng(0).integers(0, 128, size=(5, 64, 1536), dtype=np.uint8)
    a = predict(net, X, "cpu", batch=2)
    b = predict(net, X, "cpu", batch=5)
    assert a.shape == (5,) and set(a) <= {0, 1, 2}
    assert (a == b).all()          # BatchNorm uses running stats, so batching cannot change outputs
    assert not net.training
    assert predict(net, X[:0], "cpu").shape == (0,)


def test_training_learns_toy_task():
    from composer_clf.cnn.train import predict, train_cnn

    # Class 0 plays only in the low half of the roll, class 1 only in the high half.
    rng = np.random.default_rng(0)
    n = 40
    y = np.repeat([0, 1], n // 2)
    X = np.zeros((n, 64, 1536), np.uint8)
    for i in range(n):
        rows = rng.integers(0, 32, 200) + 32 * y[i]
        for r, c in zip(rows, rng.integers(0, 1526, 200)):
            X[i, r, c : c + 10] = 100
    split = np.array((["train"] * 6 + ["dev"] * 2 + ["test"] * 2) * 4)
    # 60 optimizer steps, enough for BatchNorm running statistics to settle for eval mode.
    model, history, best_epoch = train_cnn(X, y, split, n_classes=2, seed=0, augment_on=False,
                                           repeats=1, epochs=10, batch_size=4)
    assert len(history) == 10 and 1 <= best_epoch <= 10
    assert set(history[0]) == {"epoch", "train_loss", "dev_accuracy", "dev_macro_f1"}
    assert history[best_epoch - 1]["dev_macro_f1"] == max(h["dev_macro_f1"] for h in history)
    test = split == "test"
    assert (predict(model, X[test], "cpu") == y[test]).mean() >= 0.75


def _toy_catalog(tmp_path, midi_factory):
    import pandas as pd

    rows = []
    for i, (composer, split) in enumerate([("B", "train"), ("A", "train"), ("A", "dev"),
                                           ("B", "dev"), ("A", "test"), ("B", "test")] * 2):
        pitch = 40 if composer == "A" else 80
        f = midi_factory(f"{i}.mid", [(pitch, 0.0, 30.0, 90)])
        rows.append({"path": str(f), "composer": composer, "piece_id": f"{composer}:{i}", "split": split})
    pd.DataFrame(rows).to_csv(tmp_path / "cat.csv", index=False)
    return tmp_path / "cat.csv"


def test_load_rolls_builds_and_reuses_cache(tmp_path, midi_factory):
    from composer_clf.cnn.run import load_rolls

    cat = _toy_catalog(tmp_path, midi_factory)
    X, y, split, paths_, labels = load_rolls(tmp_path / "rolls.npz", cat)
    assert X.shape == (12, 64, 1536) and labels == ["A", "B"]
    assert list(y[:2]) == [1, 0] and split[2] == "dev"
    cat.write_text("broken")                       # second call must not read the catalog again
    X2, *_ = load_rolls(tmp_path / "rolls.npz", cat)
    assert (X2 == X).all()
