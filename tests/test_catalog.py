import hashlib

import pandas as pd
import pytest

from composer_clf.data.catalog import build_hf_catalog, build_maestro_catalog, dedup_key, deduplicate, normalize_composer, piece_id, select_composers


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


def test_select_composers_needs_files_and_pieces():
    rows = [_row(f"aa-piece{i}.mid", "aa", f"a{i}") for i in range(6)]            # 6 files, 6 pieces
    rows += [_row(f"bb-op1_n{i}.mid", "bb", f"b{i}") for i in range(6)]            # 6 files, 1 piece
    rows += [_row(f"cc-piece{i}.mid", "cc", f"c{i}") for i in range(3)]            # 3 files
    rows += [_row(f"unknown_artist-p{i}.mid", "unknown_artist", f"u{i}") for i in range(9)]
    kept, summary = select_composers(pd.DataFrame(rows), min_files=5, min_pieces=5)
    assert set(kept.composer) == {"aa"}
    assert summary.loc["bb", "pieces"] == 1 and not summary.loc["bb", "kept"]
    assert "unknown_artist" not in summary.index


def test_maestro_catalog(tmp_path):
    meta = pd.DataFrame({
        "canonical_composer": ["A"] * 3 + ["B"] * 1 + ["A / B"] * 2,
        "canonical_title": ["t1", "t1", "t2", "t3", "t4", "t4"],
        "split": ["train", "validation", "test", "train", "train", "test"],
        "midi_filename": [f"2004/f{i}.midi" for i in range(6)],
    })
    meta.to_csv(tmp_path / "maestro-v3.0.0.csv", index=False)
    cat = build_maestro_catalog(tmp_path, min_performances=2)
    assert list(cat.composer) == ["A", "A", "A"]
    assert list(cat.split) == ["train", "dev", "test"]
    assert cat.piece_id.iloc[0] == "A:t1"


def test_build_hf_catalog_reads_files(midi_factory):
    a = midi_factory("haendel-messiah_oratorio_hwv056-mesiah46.mid")
    midi_factory("scarlatti-k054.mid", program=6)
    cat = build_hf_catalog(a.parent)
    assert list(cat.columns) == ["path", "filename", "composer", "piece_id", "dedup_key", "md5"]
    row = cat.set_index("filename").loc[a.name]
    assert row.composer == "handel" and row.piece_id == "handel:hwv56"
    assert row.md5 == hashlib.md5(a.read_bytes()).hexdigest()
    assert len(cat) == 2


def test_merge_prefix_pieces_groups_movements_with_their_work():
    from composer_clf.data.catalog import merge_prefix_pieces

    ids = pd.Series(["handel:water_music", "handel:water_music_1_overture", "handel:water_music_allegro_2",
                     "beethoven:piano_sonata_n27", "beethoven:piano_sonata_n27_3rd_mov",
                     "haydn:piano_piece_n1", "haydn:piano_piece_n10", "mozart:water_music_x"])
    out = list(merge_prefix_pieces(ids))
    assert out[:3] == ["handel:water_music"] * 3
    assert out[3:5] == ["beethoven:piano_sonata_n27"] * 2
    assert out[5:7] == ["haydn:piano_piece_n1", "haydn:piano_piece_n10"]   # n1 is not a word prefix of n10
    assert out[7] == "mozart:water_music_x"                                 # other composer untouched


def test_merge_prefix_pieces_follows_chains():
    from composer_clf.data.catalog import merge_prefix_pieces

    out = merge_prefix_pieces(pd.Series(["a:x_y_z", "a:x", "a:x_y"]))
    assert list(out) == ["a:x", "a:x", "a:x"]
