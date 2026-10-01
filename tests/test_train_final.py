import joblib

from composer_clf import run_tabular, train_final


def test_train_final_refits_selected_lightgbm(tmp_path, synthetic_tabular):
    data = synthetic_tabular
    data.frame[["path", "composer", "piece_id"]].to_csv(tmp_path / "cat.csv", index=False)
    data.frame[["path", *data.feature_columns]].to_csv(tmp_path / "feat.csv", index=False)
    run_tabular.main(["--exp", "smoke", "--catalog", str(tmp_path / "cat.csv"),
                      "--features-csv", str(tmp_path / "feat.csv"), "--models", "lightgbm",
                      "--seeds", "0", "--out-root", str(tmp_path / "results")])
    out = tmp_path / "model.joblib"
    train_final.main(["--exp", "smoke", "--results-root", str(tmp_path / "results"), "--out", str(out)])
    bundle = joblib.load(out)
    assert bundle["feature_columns"] == data.feature_columns
    assert bundle["labels"] == ["aa", "bb", "cc"] and bundle["seed"] == 0
    assert set(bundle["pipeline"].predict(data.X)) <= {"aa", "bb", "cc"}
