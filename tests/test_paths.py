from composer_clf import paths


def test_relative_roundtrip():
    rel = "data/raw/x.mid"
    assert paths.relative(paths.resolve(rel)) == rel


def test_root_contains_pyproject():
    assert (paths.ROOT / "pyproject.toml").exists()


def test_midi_factory_writes_file(midi_factory):
    assert midi_factory("a.mid").stat().st_size > 0


def test_portable_relative_inside_absolute_outside(tmp_path):
    assert paths.portable(paths.ROOT / "data" / "x.csv") == "data/x.csv"
    assert paths.portable(tmp_path / "y.csv") == str((tmp_path / "y.csv").resolve())
