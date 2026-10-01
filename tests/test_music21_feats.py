from composer_clf.features.music21_feats import music21_features


def test_music21_features_numeric(midi_factory):
    feats = music21_features(midi_factory("a.mid"))
    assert len(feats) > 600
    assert "Identifier" not in feats


def test_extract_music21_drops_bad_file(tmp_path, midi_factory):
    from composer_clf.features.music21_feats import extract_music21

    good = midi_factory("good.mid")
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"junk")
    df, failed = extract_music21([bad, good], n_jobs=1)
    assert failed == [str(bad)]
    assert list(df["path"]) == [str(good)]
    assert df.drop(columns="path").dtypes.map(lambda t: t.kind in "fi").all()
