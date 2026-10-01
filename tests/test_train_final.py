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


def test_train_final_finds_inputs_from_another_machine(tmp_path, synthetic_tabular, monkeypatch):
    import json

    from composer_clf import paths

    data = synthetic_tabular
    processed = tmp_path / "processed"
    processed.mkdir()
    data.frame[["path", "composer", "piece_id"]].to_csv(processed / "hf_catalog.csv", index=False)
    data.frame[["path", *data.feature_columns]].to_csv(processed / "hf_jsymbolic.csv", index=False)
    run_tabular.main(["--exp", "smoke", "--models", "lightgbm", "--seeds", "0",
                      "--catalog", str(processed / "hf_catalog.csv"),
                      "--features-csv", str(processed / "hf_jsymbolic.csv"), "--out-root", str(tmp_path / "results")])
    config_file = tmp_path / "results" / "smoke" / "lightgbm" / "seed0" / "config.json"
    config = json.loads(config_file.read_text())
    config["catalog"] = "/home/someone-else/project/data/processed/hf_catalog.csv"   # recorded on the GPU box
    config["features_csv"] = "/home/someone-else/project/data/processed/hf_jsymbolic.csv"
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(paths, "DATA_PROCESSED", processed)

    train_final.main(["--exp", "smoke", "--results-root", str(tmp_path / "results"), "--out", str(tmp_path / "m.joblib")])
    assert joblib.load(tmp_path / "m.joblib")["feature_columns"] == data.feature_columns
