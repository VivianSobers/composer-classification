"""Extract features for every file in a dataset catalog."""
import argparse
from pathlib import Path

import pandas as pd

from composer_clf import paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hf", "maestro"], required=True)
    parser.add_argument("--extractor", choices=["jsymbolic", "music21"], default="jsymbolic")
    parser.add_argument("--shards", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    catalog = pd.read_csv(paths.DATA_PROCESSED / f"{args.dataset}_catalog.csv")
    files = [paths.resolve(p) for p in catalog.path]
    if args.extractor == "jsymbolic":
        from composer_clf.features.jsymbolic import extract_jsymbolic
        feats, failed = extract_jsymbolic(files, paths.DATA_PROCESSED / f"work_{args.dataset}",
                                          n_shards=args.shards, workers=args.workers)
    else:
        from composer_clf.features.music21_feats import extract_music21
        feats, failed = extract_music21(files, n_jobs=args.workers)

    feats["path"] = [paths.relative(p) for p in feats["path"]]
    stem = paths.DATA_PROCESSED / f"{args.dataset}_{args.extractor}"
    feats.to_csv(f"{stem}.csv", index=False)
    Path(f"{stem}_failed.txt").write_text("".join(f"{paths.relative(p)}\n" for p in failed))
    print(f"{len(feats)} extracted, {len(failed)} failed -> {stem}.csv")


if __name__ == "__main__":
    main()
