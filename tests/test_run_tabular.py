import json
import re

import numpy as np
import pytest
import pandas as pd

from composer_clf import run_tabular


def test_environment_records_commit_and_versions():
    env = run_tabular.environment()
    assert re.fullmatch(r"[0-9a-f]{40}", env["git_commit"])
    assert isinstance(env["git_dirty"], bool)
    assert env["packages"]["lightgbm"] == "4.7.0"
    assert env["python"].startswith("3.13")


def test_tune_and_test_picks_best_dev_config(synthetic_tabular):
    data = synthetic_tabular
    split = run_tabular.make_split("grouped", data, seed=0)
    best_params, metrics, search, preds = run_tabular.tune_and_test("knn", data, split, ["aa", "bb", "cc"], seed=0)
    assert len(search) == 8
    best_row = search.loc[search.dev_macro_f1.idxmax()]
    assert json.loads(best_row.params) == best_params and metrics["best_config"] == best_row.config
    assert metrics["dev"]["macro_f1"] == best_row.dev_macro_f1
    assert metrics["test"]["accuracy"] > 0.6
    assert list(preds.columns) == ["path", "y_true", "y_pred"] and len(preds) == (split == "test").sum()
    lo, hi = metrics["test_accuracy_ci95"]
    assert lo <= metrics["test"]["accuracy"] <= hi


def test_make_split_kinds(synthetic_tabular):
    data = synthetic_tabular
    assert set(run_tabular.make_split("random", data, 1)) == {"train", "dev", "test"}
    data.frame["split"] = "train"
    assert (run_tabular.make_split("official", data, 1) == "train").all()


def _fake_metrics(acc):
    return {"train_accuracy": 1.0, "dev": {"macro_f1": acc - 0.1},
            "test": {"accuracy": acc, "macro_f1": acc - 0.05, "balanced_accuracy": acc - 0.02}}


def test_save_run_and_summarize(tmp_path):
    for model, accs in {"knn": [0.5, 0.7], "svm": [0.8, 0.8]}.items():
        for seed, acc in enumerate(accs):
            run_tabular.save_run(tmp_path / model / f"seed{seed}", {"seed": seed, "model": model},
                                 _fake_metrics(acc), pd.DataFrame({"config": [0]}),
                                 pd.DataFrame({"path": ["x"], "y_true": ["a"], "y_pred": ["a"]}))
    assert json.loads((tmp_path / "knn" / "seed1" / "config.json").read_text())["seed"] == 1
    assert (tmp_path / "svm" / "seed0" / "predictions.csv").exists()
    summary = run_tabular.summarize(tmp_path).set_index("model")
    assert summary.loc["knn", "n_seeds"] == 2
    assert summary.loc["knn", "test_accuracy_mean"] == pytest.approx(0.6)
    assert summary.loc["svm", "test_accuracy_std"] == 0.0
    assert len(pd.read_csv(tmp_path / "runs.csv")) == 4


def test_main_writes_reproducible_runs(tmp_path, synthetic_tabular):
    data = synthetic_tabular
    data.frame[["path", "composer", "piece_id"]].to_csv(tmp_path / "cat.csv", index=False)
    data.frame[["path", *data.feature_columns]].to_csv(tmp_path / "feat.csv", index=False)
    out = tmp_path / "results"
    run_tabular.main(["--exp", "smoke", "--catalog", str(tmp_path / "cat.csv"),
                      "--features-csv", str(tmp_path / "feat.csv"), "--models", "knn", "softmax",
                      "--seeds", "0", "1", "--out-root", str(out)])
    run = out / "smoke" / "knn" / "seed1"
    config = json.loads((run / "config.json").read_text())
    assert config["seed"] == 1 and config["model"] == "knn" and "packages" in config["environment"]
    assert config["best_params"] in [{"n_neighbors": k} for k in [1, 3, 5, 7, 11, 15, 21, 31]]
    assert config["split_sizes"]["test"] > 0 and config["n_features"] == 6
    summary = pd.read_csv(out / "smoke" / "summary.csv")
    assert set(summary.model) == {"knn", "softmax"} and (summary.n_seeds == 2).all()


def test_config_stores_repo_relative_inputs(tmp_path, synthetic_tabular, monkeypatch):
    from composer_clf import paths

    data = synthetic_tabular
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    data.frame[["path", "composer", "piece_id"]].to_csv(tmp_path / "data" / "cat.csv", index=False)
    data.frame[["path", *data.feature_columns]].to_csv(tmp_path / "data" / "feat.csv", index=False)
    run_tabular.main(["--exp", "rel", "--catalog", str(tmp_path / "data" / "cat.csv"),
                      "--features-csv", str(tmp_path / "data" / "feat.csv"), "--models", "knn",
                      "--seeds", "0", "--out-root", str(tmp_path / "results")])
    config = json.loads((tmp_path / "results" / "rel" / "knn" / "seed0" / "config.json").read_text())
    assert config["catalog"] == "data/cat.csv" and config["features_csv"] == "data/feat.csv"
    assert config["out_root"] == "results"


def test_summarize_ignores_runs_outside_the_current_call(tmp_path):
    for model in ("knn", "svm"):
        for seed in range(3):
            run_tabular.save_run(tmp_path / model / f"seed{seed}", {}, _fake_metrics(0.5 + 0.1 * seed),
                                 pd.DataFrame(), pd.DataFrame())
    summary = run_tabular.summarize(tmp_path, models=["knn"], seeds=[0, 1]).set_index("model")
    assert list(summary.index) == ["knn"] and summary.loc["knn", "n_seeds"] == 2
    assert summary.loc["knn", "test_accuracy_mean"] == pytest.approx(0.55)


def test_dirty_check_ignores_result_files(monkeypatch):
    calls = []

    def fake_git(*args):
        calls.append(args)
        return "abc" if args[0] == "rev-parse" else ""

    monkeypatch.setattr(run_tabular, "_git", fake_git)
    run_tabular.environment()
    status = next(a for a in calls if a[0] == "status")
    assert ":(exclude)results" in status
