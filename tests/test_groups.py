from pathlib import Path

COLUMNS = Path(__file__).with_name("fixtures").joinpath("jsymbolic_columns.txt").read_text().split()


def test_fixture_is_full_jsymbolic_header():
    assert len(COLUMNS) == len(set(COLUMNS)) == 1495
    assert COLUMNS[0] == "Basic_Pitch_Histogram_0"
