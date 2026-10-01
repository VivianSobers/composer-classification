"""Composer labels, piece IDs and duplicate removal for both datasets."""
import re
from pathlib import Path

COMPOSER_ALIASES = {"haendel": "handel", "mendelsonn": "mendelssohn"}

_CATALOG_RE = re.compile(r"(?:^|_)(bwv|kv|k|hwv|rv|d|woo)(\d+)(?=[a-z_]|$)")
_OPUS_RE = re.compile(r"(?:^|_)op(\d+)")
_MOVEMENT_RE = re.compile(r"_?\d+mov.*$|_mov_?\d+.*$")


def normalize_composer(prefix: str) -> str:
    name = prefix.lower()
    return COMPOSER_ALIASES.get(name, name)


def normalize_segment(segment: str) -> str:
    s = segment.lower()
    s = re.sub(r"opus_?", "op", s)
    s = re.sub(r"op_(\d)", r"op\1", s)
    s = re.sub(r"(?<![a-z])no_?(\d)", r"n\1", s)
    s = re.sub(r"\d+", lambda m: str(int(m.group())), s)
    return re.sub(r"_+", "_", s).strip("_")


def split_name(filename: str) -> tuple[str, list[str]]:
    """Composer prefix and normalized name segments of a dataset filename."""
    stem = Path(filename).stem.lower()
    prefix, _, rest = stem.partition("-")
    segments = [normalize_segment(s) for s in rest.split("-")]
    return normalize_composer(prefix), [s for s in segments if s]


def piece_id(filename: str) -> str:
    """Work identifier shared by every movement or copy of one piece.

    Looks at the last two name segments for a catalogue number, then an opus
    number, else uses the last segment without its movement suffix.
    """
    composer, segments = split_name(filename)
    tail = segments[-2:][::-1]
    for segment in tail:
        if match := _CATALOG_RE.search(segment):
            kind = "k" if match.group(1) == "kv" else match.group(1)
            return f"{composer}:{kind}{match.group(2)}"
    for segment in tail:
        if match := _OPUS_RE.search(segment):
            return f"{composer}:op{match.group(1)}"
    name = _MOVEMENT_RE.sub("", segments[-1]) if segments else ""
    return f"{composer}:{name or 'unnamed'}"
