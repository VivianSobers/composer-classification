"""music21 feature extraction, used only if jSymbolic cannot be run."""
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from music21 import converter, features

_LABEL = "label"


def music21_features(path: Path) -> dict[str, float]:
    ds = features.DataSet(classLabel=_LABEL)
    ds.addFeatureExtractors(features.extractorsById("all"))
    ds.addData(converter.parse(str(path)), classValue="x", id=str(path))
    ds.process()
    names, row = ds.getAttributeLabels(), ds.getFeaturesAsList()[0]
    return {n: v for n, v in zip(names, row) if n not in ("Identifier", _LABEL)}


def _safe(path: Path):
    try:
        return music21_features(path)
    except Exception:  # music21 raises many unrelated types on unreadable files
        return None


def extract_music21(midi_paths, n_jobs: int = 4):
    """Same contract as extract_jsymbolic: (features with leading `path`, failed paths)."""
    midi_paths = [Path(p) for p in midi_paths]
    results = Parallel(n_jobs=n_jobs)(delayed(_safe)(p) for p in midi_paths)
    rows = [{"path": str(p), **r} for p, r in zip(midi_paths, results) if r is not None]
    failed = [str(p) for p, r in zip(midi_paths, results) if r is None]
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "path"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df, failed
