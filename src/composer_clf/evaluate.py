"""Classification metrics shared by every experiment."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, labels) -> dict:
    """Accuracy and macro scores; macro averages run over every label in `labels`."""
    y_true, y_pred, labels = np.asarray(y_true), np.asarray(y_pred), list(labels)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(support[i])}
                      for i, lab in enumerate(labels)},
        "labels": labels,
    }
