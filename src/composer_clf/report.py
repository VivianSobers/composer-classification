"""Collect experiment summaries into one results table."""
from pathlib import Path

import pandas as pd

from composer_clf import paths


def collect(results_dir: Path = paths.RESULTS) -> pd.DataFrame:
    """One row per experiment and model, from every results/<exp>/summary.csv."""
    frames = []
    for summary in sorted(Path(results_dir).glob("*/summary.csv")):
        df = pd.read_csv(summary)
        df.insert(0, "experiment", summary.parent.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
