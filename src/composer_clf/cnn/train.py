"""Training loop: on-the-fly augmentation, best-dev checkpoint, single test pass."""
import numpy as np
import torch

from composer_clf.cnn.model import ConvNet


def _to_tensor(batch: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(batch.astype(np.float32) / 128.0).unsqueeze(1).to(device)


@torch.no_grad()
def predict(model: ConvNet, X: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """Class indices, computed in eval mode (the original code left BatchNorm in train mode)."""
    model.eval()
    out = [model(_to_tensor(X[i : i + batch], device)).argmax(1).cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.array([], dtype=int)
