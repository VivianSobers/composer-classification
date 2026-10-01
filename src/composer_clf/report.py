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


def _fmt(mean, std) -> str:
    return f"{mean:.3f} ± {std:.3f}" if pd.notna(std) else f"{mean:.3f}"


def write_tables(results_dir: Path = paths.RESULTS) -> Path:
    """results/tables.md: test accuracy and macro-F1, mean ± std over seeds."""
    table = collect(results_dir)
    lines = ["| experiment | model | seeds | test accuracy | test macro-F1 |", "|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.experiment} | {row.model} | {row.n_seeds} | "
                     f"{_fmt(row.test_accuracy_mean, row.test_accuracy_std)} | "
                     f"{_fmt(row.test_macro_f1_mean, row.test_macro_f1_std)} |")
    out = Path(results_dir) / "tables.md"
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    print(write_tables())
