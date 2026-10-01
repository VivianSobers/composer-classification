"""Named subsets of jSymbolic feature columns."""
from pathlib import Path

_LIST_DIR = Path(__file__).with_name("group_lists")
PAPER_GROUPS = ("rhythm", "melodic", "vertical", "texture")


def _read_list(group: str) -> set[str]:
    lines = (_LIST_DIR / f"{group}.txt").read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def feature_columns(group: str, columns: list[str]) -> list[str]:
    """Columns of `columns` that belong to `group`, in their original order."""
    if group == "all":
        return list(columns)
    if group in PAPER_GROUPS:
        names = _read_list(group)
        return [c for c in columns if c in names]
    raise ValueError(f"unknown feature group: {group}")
