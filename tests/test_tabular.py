import numpy as np
import pandas as pd

from composer_clf.tabular import load_tabular


def write(tmp_path):
    cat = pd.DataFrame({
        "path": ["a.mid", "b.mid", "c.mid", "d.mid", "e.mid"],
        "composer": ["bach", "mozart", "beethoven", "chopin", "chopin"],
        "piece_id": ["bach:1", "mozart:1", "beethoven:1", "chopin:1", "chopin:2"],
    })
    feats = pd.DataFrame({
        "path": ["a.mid", "b.mid", "c.mid", "d.mid"],          # e.mid failed extraction
        "Rhythmic_Value_Histogram_0": [1.0, 2.0, 3.0, 4.0],
        "Violin_Prevalence": [0.0, 1.0, 0.0, 0.5],
        "Note_Density_per_Quarter_Note_Variability": [np.nan, 1.0, 1.0, 1.0],
        "Range": [10.0, np.nan, 30.0, 40.0],
    })
    cat.to_csv(tmp_path / "cat.csv", index=False)
    feats.to_csv(tmp_path / "f.csv", index=False)
    return tmp_path / "cat.csv", tmp_path / "f.csv"


def test_joins_catalog_with_features(tmp_path):
    data = load_tabular(*write(tmp_path))
    assert "e.mid" not in set(data.frame.path)
    assert data.X.shape[0] == len(data.y) == len(data.groups)
    assert list(data.groups) == list(data.frame.piece_id)


def test_feature_group_selection(tmp_path):
    data = load_tabular(*write(tmp_path), feature_group="instrumentation")
    assert data.feature_columns == ["Violin_Prevalence"]
