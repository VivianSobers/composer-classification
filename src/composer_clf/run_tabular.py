"""Tune, train and test tabular models; every run is written to results/."""
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from composer_clf import paths
from composer_clf.evaluate import bootstrap_ci, compute_metrics
from composer_clf.models import GRIDS, build
from composer_clf.splits import grouped_split, random_split

PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "music21"]
METRIC_COLUMNS = ["test_accuracy", "test_macro_f1", "test_balanced_accuracy", "dev_macro_f1", "train_accuracy"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()


def environment(packages=PACKAGES) -> dict:
    """Code version and library versions, stored with every run."""
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(),
            "packages": {p: metadata.version(p) for p in packages}}


def make_split(kind: str, data, seed: int) -> np.ndarray:
    if kind == "grouped":
        return grouped_split(data.y, data.groups, seed)
    if kind == "random":
        return random_split(data.y, seed)
    if kind == "official":
        return data.frame["split"].to_numpy()
    raise ValueError(f"unknown split: {kind}")


def tune_and_test(model: str, data, split: np.ndarray, labels: list[str], seed: int):
    """Fit every grid config on train, keep the best dev macro-F1, score test once."""
    X, y = data.X, data.y
    tr, dv, te = (split == "train"), (split == "dev"), (split == "test")
    rows, best = [], None
    for i, params in enumerate(GRIDS[model]):
        pipe = build(model, params, seed).fit(X[tr], y[tr])
        dev = compute_metrics(y[dv], pipe.predict(X[dv]), labels)
        rows.append({"config": i, "params": json.dumps(params), "dev_accuracy": dev["accuracy"],
                     "dev_macro_f1": dev["macro_f1"]})
        if best is None or dev["macro_f1"] > best[2]["macro_f1"]:
            best = (i, pipe, dev)
    best_i, pipe, dev = best
    test_pred = pipe.predict(X[te])
    metrics = {
        "best_config": best_i,
        "train_accuracy": float((pipe.predict(X[tr]) == y[tr]).mean()),
        "dev": dev,
        "test": compute_metrics(y[te], test_pred, labels),
        "test_accuracy_ci95": bootstrap_ci(y[te], test_pred, seed=seed),
    }
    predictions = pd.DataFrame({"path": data.frame.path[te].to_numpy(), "y_true": y[te], "y_pred": test_pred})
    return GRIDS[model][best_i], metrics, pd.DataFrame(rows), predictions


def save_run(run_dir: Path, config: dict, metrics: dict, search: pd.DataFrame, predictions: pd.DataFrame):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    search.to_csv(run_dir / "dev_search.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)


def summarize(exp_dir: Path) -> pd.DataFrame:
    """Write runs.csv (one row per model and seed) and summary.csv (mean and std per model)."""
    rows = []
    for metrics_file in sorted(Path(exp_dir).glob("*/seed*/metrics.json")):
        m = json.loads(metrics_file.read_text())
        rows.append({"model": metrics_file.parent.parent.name, "seed": int(metrics_file.parent.name[4:]),
                     "test_accuracy": m["test"]["accuracy"], "test_macro_f1": m["test"]["macro_f1"],
                     "test_balanced_accuracy": m["test"]["balanced_accuracy"],
                     "dev_macro_f1": m["dev"]["macro_f1"], "train_accuracy": m["train_accuracy"]})
    runs = pd.DataFrame(rows)
    runs.to_csv(Path(exp_dir) / "runs.csv", index=False)
    aggregations = {"n_seeds": ("seed", "nunique")}
    for col in METRIC_COLUMNS:
        aggregations[f"{col}_mean"] = (col, "mean")
        aggregations[f"{col}_std"] = (col, "std")
    summary = runs.groupby("model").agg(**aggregations).reset_index()
    summary.to_csv(Path(exp_dir) / "summary.csv", index=False)
    return summary
