"""Collect experiment summaries into one results table, and draw the result figures."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from composer_clf import paths  # noqa: E402

# Validated categorical slots (light surface) and chart chrome.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
MODEL_NAMES = {"lightgbm": "LightGBM", "softmax": "Softmax", "svm": "SVM (RBF)", "lda": "LDA",
               "knn": "k-NN", "mlp": "MLP", "cnn": "CNN"}


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


def _figure(height: float):
    fig, ax = plt.subplots(figsize=(7.5, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, ax, title: str, xlabel: str, out: Path) -> Path:
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=26)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    for label in ax.get_yticklabels():
        label.set_color(INK_SECONDARY)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _dot_plot(rows: pd.DataFrame, series: list[str], title: str, xlabel: str, out: Path) -> Path:
    """Mean +/- std dots, one row per label; rows has columns label, series, mean, std."""
    labels = list(dict.fromkeys(rows.label))
    fig, ax = _figure(0.45 * len(labels) + 1.2)
    offsets = [0.0] if len(series) == 1 else [-0.14, 0.14]
    for k, name in enumerate(series):
        part = rows[rows.series == name].set_index("label").reindex(labels)
        ys = [i + offsets[k] for i in range(len(labels))]
        ax.errorbar(part["mean"], ys, xerr=part["std"].fillna(0), fmt="o", color=SERIES[k],
                    ecolor=SERIES[k], elinewidth=1.2, capsize=0, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    if len(series) > 1:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series), fontsize=9,
                  labelcolor=INK_SECONDARY, handletextpad=0.3, borderaxespad=0.2)
    return _save(fig, ax, title, xlabel, out)


def plot_model_comparison(results_dir: Path, out: Path, exp: str = "e1_composers") -> Path:
    s = pd.read_csv(Path(results_dir) / exp / "summary.csv").sort_values("test_accuracy_mean", ascending=False)
    rows = []
    for r in s.itertuples():
        name = MODEL_NAMES.get(r.model, r.model)
        rows.append({"label": name, "series": "Test accuracy", "mean": r.test_accuracy_mean, "std": r.test_accuracy_std})
        rows.append({"label": name, "series": "Test macro-F1", "mean": r.test_macro_f1_mean, "std": r.test_macro_f1_std})
    return _dot_plot(pd.DataFrame(rows), ["Test accuracy", "Test macro-F1"],
                     "Composer classification, splits grouped by piece",
                     f"Mean ± std over {int(s.n_seeds.max())} seeds", out)


if __name__ == "__main__":
    print(write_tables())
