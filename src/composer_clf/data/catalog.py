"""Composer labels, piece IDs and duplicate removal for both datasets."""

COMPOSER_ALIASES = {"haendel": "handel", "mendelsonn": "mendelssohn"}


def normalize_composer(prefix: str) -> str:
    name = prefix.lower()
    return COMPOSER_ALIASES.get(name, name)
