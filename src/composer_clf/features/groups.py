"""Named subsets of jSymbolic feature columns."""
from pathlib import Path

_LIST_DIR = Path(__file__).with_name("group_lists")
PAPER_GROUPS = ("rhythm", "melodic", "vertical", "texture")
GROUPS = ("all", *PAPER_GROUPS, "instrumentation", "no_instrumentation")

# jSymbolic 2.2 instrumentation features (category I); histograms expand to Name_0..Name_n.
INSTRUMENTATION = {
    "Pitched_Instruments_Present", "Unpitched_Instruments_Present",
    "Note_Prevalence_of_Pitched_Instruments", "Note_Prevalence_of_Unpitched_Instruments",
    "Time_Prevalence_of_Pitched_Instruments", "Variability_of_Note_Prevalence_of_Pitched_Instruments",
    "Variability_of_Note_Prevalence_of_Unpitched_Instruments", "Number_of_Pitched_Instruments",
    "Number_of_Unpitched_Instruments", "Unpitched_Percussion_Instrument_Prevalence",
    "String_Keyboard_Prevalence", "Acoustic_Guitar_Prevalence", "Electric_Guitar_Prevalence",
    "Violin_Prevalence", "Saxophone_Prevalence", "Brass_Prevalence", "Woodwinds_Prevalence",
    "Orchestral_Strings_Prevalence", "String_Ensemble_Prevalence", "Electric_Instrument_Prevalence",
}


def _base(column: str) -> str:
    head, _, tail = column.rpartition("_")
    return head if tail.isdigit() else column


def _read_list(group: str) -> set[str]:
    lines = (_LIST_DIR / f"{group}.txt").read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def feature_columns(group: str, columns: list[str]) -> list[str]:
    """Columns of `columns` that belong to `group`, in their original order."""
    if group == "all":
        return list(columns)
    if group in ("instrumentation", "no_instrumentation"):
        want = group == "instrumentation"
        return [c for c in columns if (c in INSTRUMENTATION or _base(c) in INSTRUMENTATION) == want]
    if group in PAPER_GROUPS:
        names = _read_list(group)
        return [c for c in columns if c in names]
    raise ValueError(f"unknown feature group: {group}")
