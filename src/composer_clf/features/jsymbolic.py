"""Run jSymbolic 2.2 over many MIDI files in parallel shards."""
from pathlib import Path

import pandas as pd


def read_jsymbolic_csv(csv_path: Path) -> pd.DataFrame:
    """jSymbolic CSV output with the input file's numeric stem as `index`."""
    df = pd.read_csv(csv_path)
    source = df.columns[0]
    df["index"] = df[source].map(lambda s: int(Path(s).stem))
    return df.drop(columns=source)
