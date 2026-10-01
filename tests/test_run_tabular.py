import json
import re

import numpy as np
import pandas as pd

from composer_clf import run_tabular


def test_environment_records_commit_and_versions():
    env = run_tabular.environment()
    assert re.fullmatch(r"[0-9a-f]{40}", env["git_commit"])
    assert isinstance(env["git_dirty"], bool)
    assert env["packages"]["lightgbm"] == "4.7.0"
    assert env["python"].startswith("3.13")


def synthetic_data():
    from composer_clf.tabular import TabularData

    rng = np.random.default_rng(0)
    rows = []
    for c, name in enumerate(["aa", "bb", "cc"]):
        for g in range(10):
            for k in range(3):
                rows.append({"path": f"{name}-{g}-{k}.mid", "composer": name, "piece_id": f"{name}:{g}",
                             **{f"F_{j}": rng.normal() + 3 * c * (j == 0) for j in range(6)}})
    frame = pd.DataFrame(rows)
    cols = [f"F_{j}" for j in range(6)]
    return TabularData(X=frame[cols].to_numpy(float), y=frame.composer.to_numpy(),
                       groups=frame.piece_id.to_numpy(), frame=frame, feature_columns=cols)


def test_tune_and_test_picks_best_dev_config():
    data = synthetic_data()
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


def test_make_split_kinds():
    data = synthetic_data()
    assert set(run_tabular.make_split("random", data, 1)) == {"train", "dev", "test"}
    data.frame["split"] = "train"
    assert (run_tabular.make_split("official", data, 1) == "train").all()
