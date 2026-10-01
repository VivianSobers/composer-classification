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
