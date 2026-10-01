from composer_clf.features.music21_feats import music21_features


def test_music21_features_numeric(midi_factory):
    feats = music21_features(midi_factory("a.mid"))
    assert len(feats) > 600
    assert "Identifier" not in feats
