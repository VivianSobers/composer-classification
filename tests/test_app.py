from pathlib import Path

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

from composer_clf.models import build

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_starts_and_lists_composers(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    y = np.repeat(["bach", "chopin", "mozart"], 20)
    pipe = build("lightgbm", {"n_estimators": 10, "reg_lambda": 1.0, "learning_rate": 0.1}, seed=0)
    pipe.fit(rng.normal(size=(60, 3)), y)
    model = tmp_path / "m.joblib"
    joblib.dump({"pipeline": pipe, "feature_columns": ["a", "b", "c"], "labels": ["bach", "chopin", "mozart"]}, model)
    monkeypatch.setenv("COMPOSER_MODEL", str(model))

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Who composed this?"
    assert "bach, chopin, mozart" in at.caption[0].value
