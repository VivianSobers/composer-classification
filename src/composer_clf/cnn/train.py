"""Training loop: on-the-fly augmentation, best-dev checkpoint, single test pass."""
import copy

import numpy as np
import torch
from sklearn.metrics import f1_score

from composer_clf.cnn.model import ConvNet
from composer_clf.pianoroll import augment


def _to_tensor(batch: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(batch.astype(np.float32) / 128.0).unsqueeze(1).to(device)


@torch.no_grad()
def predict(model: ConvNet, X: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """Class indices, computed in eval mode (the original code left BatchNorm in train mode)."""
    model.eval()
    out = [model(_to_tensor(X[i : i + batch], device)).argmax(1).cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.array([], dtype=int)


def train_cnn(X, y, split, n_classes, seed, augment_on, repeats, epochs=10, lr=3e-4,
              batch_size=32, device="cpu"):
    """Train with Adam; returns the model restored to its best dev macro-F1 epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ConvNet(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_idx, dev_idx = np.where(split == "train")[0], np.where(split == "dev")[0]

    history, best_f1, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(np.repeat(train_idx, repeats))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 2:  # BatchNorm needs at least two samples
                continue
            batch = np.stack([augment(X[i], rng) if augment_on else X[i] for i in idx])
            loss = loss_fn(model(_to_tensor(batch, device)), torch.as_tensor(y[idx], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        dev_pred = predict(model, X[dev_idx], device)
        dev_f1 = f1_score(y[dev_idx], dev_pred, average="macro", labels=list(range(n_classes)), zero_division=0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "dev_accuracy": float((dev_pred == y[dev_idx]).mean()), "dev_macro_f1": float(dev_f1)})
        if dev_f1 > best_f1:
            best_f1, best_state, best_epoch = dev_f1, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return model, history, best_epoch
