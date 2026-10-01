"""Classification metrics shared by every experiment."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, labels) -> dict:
    """Accuracy and macro scores; macro averages run over every label in `labels`."""
    y_true, y_pred, labels = np.asarray(y_true), np.asarray(y_pred), list(labels)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    present = support > 0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(r[present].mean()),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(support[i])}
                      for i, lab in enumerate(labels)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def bootstrap_ci(y_true, y_pred, metric="accuracy", n_boot=1000, seed=0, alpha=0.05, labels=None):
    """Percentile bootstrap interval for one metric over resampled test items."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    if metric == "accuracy":
        stats = (y_true[idx] == y_pred[idx]).mean(axis=1)
    else:
        labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
        stats = [compute_metrics(y_true[i], y_pred[i], labels)[metric] for i in idx]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
