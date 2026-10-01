"""Train/dev/test assignment: grouped by piece (default) or random by file."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _assign(folds: list[np.ndarray], n: int) -> np.ndarray:
    split = np.full(n, "train", dtype=object)
    split[folds[0]] = "test"
    split[folds[1]] = "dev"
    return split


def _check(y: np.ndarray, split: np.ndarray) -> np.ndarray:
    for part in ("train", "dev", "test"):
        missing = set(y) - set(y[split == part])
        if missing:
            raise ValueError(f"classes missing from {part}: {sorted(missing)}")
    return split


def grouped_split(y, groups, seed: int, n_folds: int = 5) -> np.ndarray:
    """About 60/20/20, stratified by label, never splitting a group."""
    y, groups = np.asarray(y), np.asarray(groups)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [test for _, test in sgkf.split(np.zeros(len(y)), y, groups)]
    return _check(y, _assign(folds, len(y)))


def random_split(y, seed: int, n_folds: int = 5) -> np.ndarray:
    """About 60/20/20 by file, stratified by label, ignoring which piece a file belongs to."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [test for _, test in skf.split(np.zeros(len(y)), y)]
    return _check(y, _assign(folds, len(y)))
