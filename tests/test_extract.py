import pandas as pd
import pytest

from composer_clf import paths
from composer_clf.features import extract


@pytest.mark.slow
@pytest.mark.skipif(not paths.JSYMBOLIC_JAR.exists(), reason="jSymbolic not downloaded")
def test_extract_cli_writes_relative_paths(tmp_path, midi_factory, monkeypatch):
    midi = midi_factory("bach-x.mid")
    bad = tmp_path / "bach-bad.mid"
    bad.write_bytes(b"junk")
    jar, config = paths.JSYMBOLIC_JAR, paths.JSYMBOLIC_CONFIG
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATA_PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "JSYMBOLIC_JAR", jar)
    monkeypatch.setattr(paths, "JSYMBOLIC_CONFIG", config)
    (tmp_path / "processed").mkdir()
    pd.DataFrame({"path": [midi.name, bad.name]}).to_csv(tmp_path / "processed" / "hf_catalog.csv", index=False)

    extract.main(["--dataset", "hf", "--shards", "1", "--workers", "1"])

    feats = pd.read_csv(tmp_path / "processed" / "hf_jsymbolic.csv")
    assert list(feats.path) == ["bach-x.mid"]
    assert (tmp_path / "processed" / "hf_jsymbolic_failed.txt").read_text() == "bach-bad.mid\n"


@pytest.mark.slow
@pytest.mark.skipif(not paths.JSYMBOLIC_JAR.exists(), reason="jSymbolic not downloaded")
def test_extract_cli_resume_keeps_earlier_work(tmp_path, midi_factory, monkeypatch):
    midi = midi_factory("bach-x.mid")
    jar, config = paths.JSYMBOLIC_JAR, paths.JSYMBOLIC_CONFIG
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATA_PROCESSED", tmp_path / "processed")
    (tmp_path / "processed").mkdir()
    pd.DataFrame({"path": [midi.name]}).to_csv(tmp_path / "processed" / "hf_catalog.csv", index=False)
    extract.main(["--dataset", "hf", "--shards", "1", "--workers", "1"])
    work = tmp_path / "processed" / "work_hf"
    before = sorted(p.name for p in work.iterdir())

    extract.main(["--dataset", "hf", "--shards", "1", "--workers", "1", "--resume"])

    assert sorted(p.name for p in work.iterdir()) == before
    assert list(pd.read_csv(tmp_path / "processed" / "hf_jsymbolic.csv").path) == ["bach-x.mid"]
