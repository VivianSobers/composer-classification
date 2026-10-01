import pandas as pd

from composer_clf.report import collect, write_tables


def _write_summaries(tmp_path):
    for exp in ("e2_random_split", "e1_composers"):
        (tmp_path / exp).mkdir()
        pd.DataFrame({"model": ["lightgbm"], "n_seeds": [5], "test_accuracy_mean": [0.7],
                      "test_accuracy_std": [0.02], "test_macro_f1_mean": [0.6], "test_macro_f1_std": [0.03]}
                     ).to_csv(tmp_path / exp / "summary.csv", index=False)
    (tmp_path / "figures").mkdir()


def test_collect_orders_experiments(tmp_path):
    _write_summaries(tmp_path)
    table = collect(tmp_path)
    assert list(table.experiment) == ["e1_composers", "e2_random_split"]
    assert list(table.columns[:2]) == ["experiment", "model"]


def test_write_tables_formats_mean_and_std(tmp_path):
    _write_summaries(tmp_path)
    text = write_tables(tmp_path).read_text()
    assert text.splitlines()[0].startswith("| experiment | model | seeds |")
    assert "| e1_composers | lightgbm | 5 | 0.700 ± 0.020 | 0.600 ± 0.030 |" in text


def _fake_exp(root, exp, models, acc=0.7, seeds=(0, 1)):
    import json

    for model in models:
        for seed in seeds:
            d = root / exp / model / f"seed{seed}"
            d.mkdir(parents=True)
            m = {"train_accuracy": 1.0, "dev": {"macro_f1": acc},
                 "test": {"accuracy": acc + 0.01 * seed, "macro_f1": acc - 0.1, "balanced_accuracy": acc,
                          "labels": ["a", "b", "c"], "confusion": [[5, 1, 0], [2, 3, 1], [0, 0, 4]]}}
            (d / "metrics.json").write_text(json.dumps(m))
            pd.DataFrame({"epoch": [1, 2, 3], "train_loss": [1.0, 0.5, 0.2],
                          "dev_accuracy": [0.3, 0.5, 0.6], "dev_macro_f1": [0.2, 0.4, 0.5]}
                         ).to_csv(d / "dev_search.csv", index=False)
    from composer_clf.run_tabular import summarize
    summarize(root / exp)


def _is_png(path):
    return path.exists() and path.read_bytes()[:4] == b"\x89PNG"


def test_plot_model_comparison(tmp_path):
    from composer_clf.report import plot_model_comparison

    _fake_exp(tmp_path, "e1_composers", ["lightgbm", "svm", "knn"])
    assert _is_png(plot_model_comparison(tmp_path, tmp_path / "fig.png"))


def test_plot_leakage(tmp_path):
    from composer_clf.report import plot_leakage

    _fake_exp(tmp_path, "e1_composers", ["lightgbm", "svm", "knn"], acc=0.6)
    _fake_exp(tmp_path, "e2_random_split", ["lightgbm", "svm"], acc=0.8)
    assert _is_png(plot_leakage(tmp_path, tmp_path / "fig.png"))


def test_plot_confusion(tmp_path):
    from composer_clf.report import plot_confusion

    _fake_exp(tmp_path, "e1_composers", ["lightgbm"])
    assert _is_png(plot_confusion(tmp_path, tmp_path / "fig.png"))


def test_plot_ablation(tmp_path):
    from composer_clf.report import plot_ablation

    _fake_exp(tmp_path, "e1_composers", ["lightgbm"])
    for g in ("rhythm", "melodic", "instrumentation"):
        _fake_exp(tmp_path, f"e3_composer_{g}", ["lightgbm"], acc=0.5)
    for g in ("all", "rhythm"):
        _fake_exp(tmp_path, f"e3_era_{g}", ["lightgbm"], acc=0.8)
    assert _is_png(plot_ablation(tmp_path, tmp_path / "fig.png"))


def test_plot_importance(tmp_path):
    import joblib
    import numpy as np

    from composer_clf.models import build
    from composer_clf.report import plot_importance

    rng = np.random.default_rng(0)
    y = np.repeat(["a", "b", "c"], 20)
    X = rng.normal(size=(60, 25)) + (y == "b")[:, None]
    pipe = build("lightgbm", {"n_estimators": 20, "reg_lambda": 1.0, "learning_rate": 0.1}, seed=0).fit(X, y)
    joblib.dump({"pipeline": pipe, "feature_columns": [f"Feature_{i}" for i in range(25)]}, tmp_path / "m.joblib")
    assert _is_png(plot_importance(tmp_path / "m.joblib", tmp_path / "fig.png", top=20))


def test_plot_maestro_comparison(tmp_path):
    from composer_clf.report import plot_maestro_comparison

    _fake_exp(tmp_path, "e4_maestro_tabular", ["lightgbm", "svm"])
    _fake_exp(tmp_path, "e5_cnn_noaug", ["cnn"], acc=0.5)
    _fake_exp(tmp_path, "e5_cnn_aug10", ["cnn"], acc=0.55)
    assert _is_png(plot_maestro_comparison(tmp_path, tmp_path / "fig.png"))


def test_plot_cnn_curves(tmp_path):
    from composer_clf.report import plot_cnn_curves

    _fake_exp(tmp_path, "e5_cnn_noaug", ["cnn"])
    _fake_exp(tmp_path, "e5_cnn_aug10", ["cnn"])
    assert _is_png(plot_cnn_curves(tmp_path, tmp_path / "fig.png"))


def test_make_figures_skips_missing_experiments(tmp_path):
    from composer_clf.report import make_figures

    _fake_exp(tmp_path, "e1_composers", ["lightgbm", "svm"])
    made = make_figures(tmp_path, bundle_path=tmp_path / "missing.joblib")
    assert sorted(p.name for p in made) == ["confusion_lightgbm.png", "model_comparison.png"]
    assert all(_is_png(p) for p in made)
