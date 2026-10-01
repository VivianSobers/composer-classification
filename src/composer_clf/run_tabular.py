"""Tune, train and test tabular models; every run is written to results/."""
import argparse
import json
import platform
import subprocess
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from composer_clf import paths
from composer_clf.evaluate import bootstrap_ci, compute_metrics
from composer_clf.models import GRIDS, MODEL_NAMES, build
from composer_clf.splits import grouped_split, random_split
from composer_clf.tabular import load_tabular

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


def portable(p) -> str:
    """Repo-relative path for files inside the repo, so configs stay valid on other machines."""
    p = Path(p).resolve()
    return paths.relative(p) if p.is_relative_to(paths.ROOT) else str(p)


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


def main(argv=None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--dataset", choices=["hf", "maestro"], default="hf")
    parser.add_argument("--extractor", choices=["jsymbolic", "music21"], default="jsymbolic")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--split", choices=["grouped", "random", "official"], default="grouped")
    parser.add_argument("--features", default="all")
    parser.add_argument("--task", choices=["composer", "era"], default="composer")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-root", type=Path, default=paths.RESULTS)
    args = parser.parse_args(argv)

    catalog = args.catalog or paths.DATA_PROCESSED / f"{args.dataset}_catalog.csv"
    features_csv = args.features_csv or paths.DATA_PROCESSED / f"{args.dataset}_{args.extractor}.csv"
    data = load_tabular(catalog, features_csv, args.features, args.task)
    labels = sorted(set(data.y))
    exp_dir = args.out_root / args.exp
    env = environment()
    warnings.filterwarnings("ignore", message=".*converge.*")
    warnings.filterwarnings("ignore", message=".*least populated class.*")

    for seed in args.seeds:
        split = make_split(args.split, data, seed)
        for model in args.models:
            best_params, metrics, search, predictions = tune_and_test(model, data, split, labels, seed)
            config = {**vars(args), "catalog": portable(catalog), "features_csv": portable(features_csv),
                      "out_root": portable(args.out_root), "seed": seed, "model": model,
                      "grid": GRIDS[model], "best_params": best_params, "labels": labels,
                      "n_features": len(data.feature_columns),
                      "split_sizes": {s: int((split == s).sum()) for s in ("train", "dev", "test")},
                      "environment": env}
            save_run(exp_dir / model / f"seed{seed}", config, metrics, search, predictions)
            print(f"{args.exp} seed={seed} {model}: dev macro-F1 {metrics['dev']['macro_f1']:.3f} "
                  f"test acc {metrics['test']['accuracy']:.3f}", flush=True)
    summary = summarize(exp_dir)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
