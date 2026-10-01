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
# Sequential blue ramp, steps 100 to 700.
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
         "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
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


def plot_leakage(results_dir: Path, out: Path, grouped: str = "e1_composers", random: str = "e2_random_split") -> Path:
    """Test accuracy with piece-grouped vs random file splits, for the models run in both."""
    g = pd.read_csv(Path(results_dir) / grouped / "summary.csv").set_index("model")
    r = pd.read_csv(Path(results_dir) / random / "summary.csv").set_index("model")
    rows = []
    for model in r.index.intersection(g.index):
        name = MODEL_NAMES.get(model, model)
        rows.append({"label": name, "series": "Grouped by piece", "mean": g.loc[model, "test_accuracy_mean"],
                     "std": g.loc[model, "test_accuracy_std"]})
        rows.append({"label": name, "series": "Random by file", "mean": r.loc[model, "test_accuracy_mean"],
                     "std": r.loc[model, "test_accuracy_std"]})
    return _dot_plot(pd.DataFrame(rows), ["Grouped by piece", "Random by file"],
                     "Test accuracy under two ways of splitting the data", "Test accuracy, mean ± std over seeds", out)


def plot_confusion(results_dir: Path, out: Path, exp: str = "e1_composers", model: str = "lightgbm",
                   seed: int = 0) -> Path:
    """Row-normalized test confusion matrix; only the diagonal (per-composer recall) is labeled."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    m = json.loads((Path(results_dir) / exp / model / f"seed{seed}" / "metrics.json").read_text())["test"]
    counts = np.array(m["confusion"], dtype=float)
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    labels = [name.capitalize() for name in m["labels"]]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.42 * n + 2.6, 0.42 * n + 1.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    mesh = ax.pcolormesh(share, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    for i in range(n):
        ax.text(i + 0.5, i + 0.5, f"{share[i, i]:.0%}", ha="center", va="center", fontsize=7,
                color="#ffffff" if share[i, i] > 0.45 else INK)
    ax.set_xticks([i + 0.5 for i in range(n)], labels, rotation=60, ha="right")
    ax.set_yticks([i + 0.5 for i in range(n)], labels)
    ax.invert_yaxis()
    ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("True composer", color=INK_SECONDARY, fontsize=9)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02, format=matplotlib.ticker.PercentFormatter(1.0))
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, length=0, labelsize=8)
    bar.set_label("Share of the true composer's test pieces", color=INK_SECONDARY, fontsize=8)
    ax.set_title(f"{MODEL_NAMES.get(model, model)} test predictions (seed {seed})", loc="left", color=INK,
                 fontsize=11, pad=12)
    ax.set_xlabel("Predicted composer", color=INK_SECONDARY, fontsize=9)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


GROUP_NAMES = {"all": "All features", "rhythm": "Rhythm", "melodic": "Melodic intervals",
               "vertical": "Vertical intervals", "texture": "Texture", "instrumentation": "Instrumentation",
               "no_instrumentation": "All but instrumentation"}


def plot_ablation(results_dir: Path, out: Path) -> Path:
    """LightGBM test accuracy per feature group, for the composer task and the era task."""
    results_dir = Path(results_dir)
    sources = {"Composer task": lambda g: "e1_composers" if g == "all" else f"e3_composer_{g}",
               "Era task": lambda g: f"e3_era_{g}"}
    rows = []
    for group, label in GROUP_NAMES.items():
        for series, exp_of in sources.items():
            summary = results_dir / exp_of(group) / "summary.csv"
            if summary.exists():
                r = pd.read_csv(summary).set_index("model").loc["lightgbm"]
                rows.append({"label": label, "series": series, "mean": r.test_accuracy_mean,
                             "std": r.test_accuracy_std})
    return _dot_plot(pd.DataFrame(rows), list(sources), "LightGBM accuracy by feature group",
                     "Test accuracy, mean ± std over seeds", out)


def plot_importance(bundle_path: Path, out: Path, top: int = 20) -> Path:
    """Top LightGBM features by total split gain, from the saved demo model."""
    import joblib

    bundle = joblib.load(bundle_path)
    booster = bundle["pipeline"].named_steps["clf"].booster_
    gain = pd.Series(booster.feature_importance(importance_type="gain"), index=bundle["feature_columns"])
    gain = gain.sort_values(ascending=False).head(top)
    fig, ax = _figure(0.28 * len(gain) + 1.2)
    ax.barh(range(len(gain)), gain.to_numpy(), height=0.55, color=SERIES[0])
    ax.set_yticks(range(len(gain)), [name.replace("_", " ") for name in gain.index], fontsize=8)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    return _save(fig, ax, f"Top {len(gain)} features of the LightGBM model", "Total information gain", out)


CNN_RUNS = {"e5_cnn_noaug": "CNN", "e5_cnn_aug10": "CNN + augmentation"}


def plot_maestro_comparison(results_dir: Path, out: Path) -> Path:
    """Test accuracy on MAESTRO's official split: jSymbolic models (E4) next to the piano-roll CNN (E5)."""
    results_dir = Path(results_dir)
    rows = []
    for r in pd.read_csv(results_dir / "e4_maestro_tabular" / "summary.csv").itertuples():
        rows.append({"label": MODEL_NAMES.get(r.model, r.model), "series": "Test accuracy",
                     "mean": r.test_accuracy_mean, "std": r.test_accuracy_std})
    for exp, label in CNN_RUNS.items():
        summary = results_dir / exp / "summary.csv"
        if summary.exists():
            r = pd.read_csv(summary).iloc[0]
            rows.append({"label": label, "series": "Test accuracy", "mean": r.test_accuracy_mean,
                         "std": r.test_accuracy_std})
    rows = pd.DataFrame(rows).sort_values("mean", ascending=False)
    return _dot_plot(rows, ["Test accuracy"], "MAESTRO, 6 composers, official split",
                     "Test accuracy, mean ± std over seeds", out)


if __name__ == "__main__":
    print(write_tables())
