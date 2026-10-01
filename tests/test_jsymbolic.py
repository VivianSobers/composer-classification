import pytest

from composer_clf import paths
from composer_clf.features.jsymbolic import read_jsymbolic_csv


def test_read_jsymbolic_csv_maps_index(tmp_path):
    csv = tmp_path / "v.csv"
    csv.write_text(",Feat_A,Feat_B\n/tmp/w/shard00/00003.mid,1.0,2.0\n/tmp/w/shard01/00010.midi,3.0,4.0\n")
    df = read_jsymbolic_csv(csv)
    assert list(df["index"]) == [3, 10]
    assert list(df.columns) == ["Feat_A", "Feat_B", "index"]
