"""Train/dev/test assignment: grouped by piece (default) or random by file."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _assign(folds: list[np.ndarray], n: int) -> np.ndarray:
    split = np.full(n, "train", dtype=object)
    split[folds[0]] = "test"
    split[folds[1]] = "dev"
    return split


def grouped_split(y, groups, seed: int, n_folds: int = 5) -> np.ndarray:
    """About 60/20/20, stratified by label, never splitting a group."""
    y, groups = np.asarray(y), np.asarray(groups)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [test for _, test in sgkf.split(np.zeros(len(y)), y, groups)]
    return _assign(folds, len(y))
