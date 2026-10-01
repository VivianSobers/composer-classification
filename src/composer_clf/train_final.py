"""Fit the demo LightGBM model with the configuration an experiment selected for one seed."""
import argparse
import json
from pathlib import Path

import joblib

from composer_clf import paths
from composer_clf.models import build
from composer_clf.run_tabular import make_split
from composer_clf.tabular import load_tabular


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="e1_composers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-root", type=Path, default=paths.RESULTS)
    parser.add_argument("--out", type=Path, default=paths.MODELS / "lightgbm_hf.joblib")
    args = parser.parse_args(argv)

    run_dir = args.results_root / args.exp / "lightgbm" / f"seed{args.seed}"
    config = json.loads((run_dir / "config.json").read_text())
    data = load_tabular(config["catalog"], config["features_csv"], config["features"], config["task"])
    split = make_split(config["split"], data, args.seed)
    train = split == "train"
    pipe = build("lightgbm", config["best_params"], args.seed).fit(data.X[train], data.y[train])
    bundle = {"pipeline": pipe, "feature_columns": data.feature_columns, "labels": sorted(set(data.y)),
              "params": config["best_params"], "seed": args.seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"saved {args.out} ({int(train.sum())} training pieces)")


if __name__ == "__main__":
    main()
