from composer_clf import paths


def test_relative_roundtrip():
    rel = "data/raw/x.mid"
    assert paths.relative(paths.resolve(rel)) == rel


def test_root_contains_pyproject():
    assert (paths.ROOT / "pyproject.toml").exists()
