import pytest

from composer_clf import paths
from composer_clf.features.jsymbolic import read_jsymbolic_csv


def test_read_jsymbolic_csv_maps_index(tmp_path):
    csv = tmp_path / "v.csv"
    csv.write_text(",Feat_A,Feat_B\n/tmp/w/shard00/00003.mid,1.0,2.0\n/tmp/w/shard01/00010.midi,3.0,4.0\n")
    df = read_jsymbolic_csv(csv)
    assert list(df["index"]) == [3, 10]
    assert list(df.columns) == ["Feat_A", "Feat_B", "index"]


needs_jar = pytest.mark.skipif(not paths.JSYMBOLIC_JAR.exists(), reason="jSymbolic not downloaded")


@pytest.mark.slow
@needs_jar
def test_extract_real_files(tmp_path, midi_factory):
    from composer_clf.features.jsymbolic import extract_jsymbolic

    files = [midi_factory("a.mid"), midi_factory("b.mid", program=40), midi_factory("c.mid", program=6)]
    feats, failed = extract_jsymbolic(files, tmp_path / "work", n_shards=2, workers=2)
    assert failed == []
    assert list(feats["path"]) == [str(f) for f in files]
    assert feats.shape[1] == 1 + 1495
