from pathlib import Path

import pytest

from composer_clf.features.groups import feature_columns

COLUMNS = Path(__file__).with_name("fixtures").joinpath("jsymbolic_columns.txt").read_text().split()


def test_fixture_is_full_jsymbolic_header():
    assert len(COLUMNS) == len(set(COLUMNS)) == 1495
    assert COLUMNS[0] == "Basic_Pitch_Histogram_0"



@pytest.mark.parametrize("group, size", [
    ("all", 1495), ("rhythm", 255), ("melodic", 146), ("vertical", 46), ("texture", 24),
])
def test_paper_group_sizes(group, size):
    assert len(feature_columns(group, COLUMNS)) == size


def test_groups_keep_column_order():
    cols = feature_columns("melodic", COLUMNS)
    assert cols == [c for c in COLUMNS if c in set(cols)]


def test_unknown_group():
    with pytest.raises(ValueError):
        feature_columns("harmony", COLUMNS)


@pytest.mark.parametrize("group, size", [("instrumentation", 493), ("no_instrumentation", 1495 - 493)])
def test_instrumentation_split(group, size):
    assert len(feature_columns(group, COLUMNS)) == size


def test_instrumentation_includes_histograms_and_scalars():
    cols = feature_columns("instrumentation", COLUMNS)
    assert "Violin_Prevalence" in cols and "Pitched_Instruments_Present_127" in cols
    assert "Pitch_Class_Histogram_0" not in cols
