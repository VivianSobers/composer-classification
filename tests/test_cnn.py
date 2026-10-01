import numpy as np
import torch

from composer_clf.cnn.model import ConvNet


def test_forward_shape_and_flat_size():
    net = ConvNet(6)
    assert net.FLAT == 6144
    assert net(torch.zeros(2, 1, 64, 1536)).shape == (2, 6)
