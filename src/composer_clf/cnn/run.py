"""E5: piano-roll CNN on MAESTRO's official split."""
from pathlib import Path

import numpy as np
import pandas as pd

from composer_clf import paths
from composer_clf.pianoroll import midi_to_roll

CACHE = paths.DATA_PROCESSED / "maestro_rolls.npz"
CATALOG = paths.DATA_PROCESSED / "maestro_catalog.csv"


def load_rolls(cache: Path = CACHE, catalog_csv: Path = CATALOG):
    """Rolls, label indices, splits, paths and label names; built once, then read from cache."""
    if not Path(cache).exists():
        cat = pd.read_csv(catalog_csv)
        labels = sorted(cat.composer.unique())
        X = np.stack([midi_to_roll(paths.resolve(p)) for p in cat.path])
        y = cat.composer.map({c: i for i, c in enumerate(labels)}).to_numpy()
        np.savez_compressed(cache, X=X, y=y, split=cat.split.to_numpy(), path=cat.path.to_numpy(),
                            labels=np.array(labels))
    data = np.load(cache, allow_pickle=True)
    return data["X"], data["y"], data["split"], data["path"], [str(s) for s in data["labels"]]
