"""E5: piano-roll CNN on MAESTRO's official split."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from composer_clf import paths
from composer_clf.cnn.train import predict, train_cnn
from composer_clf.evaluate import bootstrap_ci, compute_metrics
from composer_clf.pianoroll import midi_to_roll
from composer_clf.run_tabular import environment, save_run, summarize

CACHE = paths.DATA_PROCESSED / "maestro_rolls.npz"
CATALOG = paths.DATA_PROCESSED / "maestro_catalog.csv"


def catalog_digest(catalog_csv: Path) -> str:
    return hashlib.md5(Path(catalog_csv).read_bytes()).hexdigest()


def load_rolls(cache: Path = CACHE, catalog_csv: Path = CATALOG):
    """Rolls, label indices, splits, paths and label names, cached until the catalog changes."""
    digest = catalog_digest(catalog_csv)
    if Path(cache).exists():
        data = np.load(cache, allow_pickle=True)
        if "catalog_md5" in data.files and str(data["catalog_md5"]) == digest:
            return data["X"], data["y"], data["split"], data["path"], [str(s) for s in data["labels"]]
    cat = pd.read_csv(catalog_csv)
    labels = sorted(cat.composer.unique())
    X = np.stack([midi_to_roll(paths.resolve(p)) for p in cat.path])
    y = cat.composer.map({c: i for i, c in enumerate(labels)}).to_numpy()
    np.savez_compressed(cache, X=X, y=y, split=cat.split.to_numpy(), path=cat.path.to_numpy(),
                        labels=np.array(labels), catalog_md5=np.array(digest))
    return X, y, cat.split.to_numpy(), cat.path.to_numpy(), labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-root", type=Path, default=paths.RESULTS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="auto uses the GPU when one is visible; cpu keeps a shared GPU free")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 keeps its default)")
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    X, y, split, file_paths, labels = load_rolls(CACHE, CATALOG)
    env = environment(["numpy", "torch", "pretty_midi"])
    names = np.array(labels)
    for seed in args.seeds:
        model, history, best_epoch = train_cnn(X, y, split, len(labels), seed, args.augment, args.repeats,
                                               args.epochs, args.lr, args.batch_size, device)
        sets = {s: split == s for s in ("train", "dev", "test")}
        preds = {s: names[predict(model, X[m], device)] for s, m in sets.items()}
        truth = {s: names[y[m]] for s, m in sets.items()}
        metrics = {"best_epoch": best_epoch,
                   "train_accuracy": float((preds["train"] == truth["train"]).mean()),
                   "dev": compute_metrics(truth["dev"], preds["dev"], labels),
                   "test": compute_metrics(truth["test"], preds["test"], labels),
                   "test_accuracy_ci95": bootstrap_ci(truth["test"], preds["test"], seed=seed)}
        config = {**vars(args), "out_root": paths.portable(args.out_root), "seed": seed, "model": "cnn",
                  "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                  "torch_threads": torch.get_num_threads(),
                  "catalog": paths.portable(CATALOG), "catalog_md5": catalog_digest(CATALOG),
                  "labels": labels, "split_sizes": {s: int(m.sum()) for s, m in sets.items()},
                  "environment": env}
        # For the CNN the dev search is the per-epoch selection, so the history goes in dev_search.csv.
        save_run(args.out_root / args.exp / "cnn" / f"seed{seed}", config, metrics, pd.DataFrame(history),
                 pd.DataFrame({"path": file_paths[sets["test"]], "y_true": truth["test"], "y_pred": preds["test"]}))
        print(f"{args.exp} seed={seed}: best epoch {best_epoch}, test acc {metrics['test']['accuracy']:.3f}",
              flush=True)
    print(summarize(args.out_root / args.exp, models=["cnn"], seeds=args.seeds).to_string(index=False))


if __name__ == "__main__":
    main()
