from composer_clf.data.catalog import normalize_composer


def test_aliases():
    assert normalize_composer("haendel") == "handel"
    assert normalize_composer("Mendelsonn") == "mendelssohn"
    assert normalize_composer("chopin") == "chopin"
