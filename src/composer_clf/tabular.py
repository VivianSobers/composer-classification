"""Join a catalog with its feature table and turn it into model inputs."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from composer_clf.features.groups import feature_columns

# Mostly NaN in jSymbolic output; the original paper dropped it too.
DROP_FEATURES = ["Note_Density_per_Quarter_Note_Variability"]


@dataclass
class TabularData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    frame: pd.DataFrame
    feature_columns: list[str]


def load_tabular(catalog_csv: Path, features_csv: Path, feature_group: str = "all",
                 task: str = "composer") -> TabularData:
    catalog = pd.read_csv(catalog_csv)
    feats = pd.read_csv(features_csv)
    available = [c for c in feats.columns if c != "path" and c not in DROP_FEATURES]
    cols = feature_columns(feature_group, available)
    df = catalog.merge(feats[["path", *cols]], on="path", how="inner")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df[~df[cols].isna().any(axis=1)]
    df = df.assign(label=df.composer).reset_index(drop=True)
    return TabularData(X=df[cols].to_numpy(float), y=df.label.to_numpy(),
                       groups=df.piece_id.to_numpy(), frame=df, feature_columns=cols)
