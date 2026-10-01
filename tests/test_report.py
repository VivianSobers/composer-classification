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
