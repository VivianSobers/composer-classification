import pandas as pd
import pytest

from composer_clf.data.catalog import dedup_key, deduplicate, normalize_composer, piece_id


def test_aliases():
    assert normalize_composer("haendel") == "handel"
    assert normalize_composer("Mendelsonn") == "mendelssohn"
    assert normalize_composer("chopin") == "chopin"


@pytest.mark.parametrize("filename, expected", [
    ("bach-bwv001-_400_chorales-028500b.mid", "bach:28500b"),
    ("bach-bwv001-_400_chorales-bwv768_chorale_and_variations-bsgjg_c.mid", "bach:bwv768"),
    ("bach-bwv772-786_two_part_inventions-bwv777_invention_n06.mid", "bach:bwv777"),
    ("scarlatti-sonatas-k171.mid", "scarlatti:k171"),
    ("scarlatti-k054.mid", "scarlatti:k54"),
    ("mozart-symphonies-symphony_n31_k297_2mov_paris.mid", "mozart:k297"),
    ("haendel-messiah_oratorio_hwv056-mesiah46.mid", "handel:hwv56"),
    ("haendel-concertos_grossos_hwv319-330_op06-concerto_grosso_op6_n06_4mov.mid", "handel:op6"),
    ("vivaldi-concertos_op3_01-12_lestro_armonico-concerto_op3_n11_for_2_violins_rv565_2mov.mid", "vivaldi:rv565"),
    ("clementi-sonatina_opus36_no3.mid", "clementi:op36"),
    ("clementi-sonatina_op36_n6_1mov.mid", "clementi:op36"),
    ("haydn-symphony_n55_4mov_der_schulmeister.mid", "haydn:symphony_n55"),
    ("schubert-piano_sonatas-piano_sonata_n21_d960_2mov.mid", "schubert:d960"),
    ("beethoven-symphonies-symphony_op125_n9_4mov_choral.mid", "beethoven:op125"),
    ("maier-atalanta_fugiens_no16.mid", "maier:atalanta_fugiens_n16"),
    ("burgmuller-la_bergeronette_op_100_no_11.mid", "burgmuller:op100"),
])
def test_piece_id(filename, expected):
    assert piece_id(filename) == expected


def test_movements_of_one_work_share_piece_id():
    assert piece_id("haydn-symphony_n55_1mov.mid") == piece_id("haydn-symphony_n55_4mov_der_schulmeister.mid")


def test_dedup_key_matches_renamed_copies():
    assert dedup_key("scarlatti-sonatas-k171.mid") == dedup_key("scarlatti-k171.mid")
    assert dedup_key("clementi-sonatina_op36_n3_1mov.mid") != dedup_key("clementi-sonatina_op36_n3_2mov.mid")


def _row(name, composer, md5):
    return {"path": f"data/raw/{name}", "filename": name, "composer": composer,
            "piece_id": piece_id(name), "dedup_key": dedup_key(name), "md5": md5}


def test_deduplicate_drops_md5_and_name_duplicates():
    df = pd.DataFrame([
        _row("scarlatti-sonatas-k171.mid", "scarlatti", "a"),
        _row("scarlatti-k171.mid", "scarlatti", "b"),          # same dedup key
        _row("scarlatti-sonatas-k172.mid", "scarlatti", "c"),
        _row("scarlatti-sonatas-k173.mid", "scarlatti", "c"),  # same bytes, same composer
    ])
    out = deduplicate(df)
    assert sorted(out.filename) == ["scarlatti-sonatas-k171.mid", "scarlatti-sonatas-k172.mid"]


def test_deduplicate_drops_files_claimed_by_two_composers():
    df = pd.DataFrame([
        _row("bach-x.mid", "bach", "same"),
        _row("handel-y.mid", "handel", "same"),
        _row("bach-z.mid", "bach", "other"),
    ])
    assert list(deduplicate(df).filename) == ["bach-z.mid"]
