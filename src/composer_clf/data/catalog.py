"""Composer labels, piece IDs and duplicate removal for both datasets."""
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

from composer_clf import paths

COMPOSER_ALIASES = {"haendel": "handel", "mendelsonn": "mendelssohn"}
EXCLUDED_COMPOSERS = {"unknown_artist"}
MAESTRO_SPLITS = {"train": "train", "validation": "dev", "test": "test"}

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


def merge_prefix_pieces(piece_ids: pd.Series) -> pd.Series:
    """Map each piece ID onto the shortest same-composer ID that is a word prefix of it.

    Catches movement names the parser misses ("water_music_1_overture" -> "water_music").
    Over-merging distinct works only makes the split stricter, never leakier.
    """
    unique = sorted(set(piece_ids), key=len)
    root = {}
    for pid in unique:
        root[pid] = next((root[shorter] for shorter in unique
                          if len(shorter) < len(pid) and pid.startswith(shorter + "_")), pid)
    return piece_ids.map(root)


def build_hf_catalog(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(raw_dir).glob("*.mid*")):
        composer, _ = split_name(p.name)
        rows.append({"path": paths.portable(p), "filename": p.name, "composer": composer,
                     "piece_id": piece_id(p.name), "dedup_key": dedup_key(p.name),
                     "md5": hashlib.md5(p.read_bytes()).hexdigest()})
    return pd.DataFrame(rows)


def dedup_key(filename: str) -> str:
    """Composer plus the normalized last name segment; renamed copies collide."""
    composer, segments = split_name(filename)
    return f"{composer}:{segments[-1] if segments else ''}"


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop byte-identical files claimed by several composers, then repeat copies."""
    composers_per_md5 = df.groupby("md5").composer.nunique()
    ambiguous = composers_per_md5.index[composers_per_md5 > 1]
    df = df[~df.md5.isin(ambiguous)]
    df = df.drop_duplicates("md5", keep="first")
    return df.drop_duplicates("dedup_key", keep="first").reset_index(drop=True)


def select_composers(df: pd.DataFrame, min_files: int = 50, min_pieces: int = 5):
    """Keep composers with enough files, and enough distinct pieces for a grouped split."""
    df = df[~df.composer.isin(EXCLUDED_COMPOSERS)]
    summary = df.groupby("composer").agg(files=("path", "size"), pieces=("piece_id", "nunique"))
    summary["kept"] = (summary.files >= min_files) & (summary.pieces >= min_pieces)
    kept = summary.index[summary.kept]
    return (df[df.composer.isin(kept)].reset_index(drop=True),
            summary.sort_values("files", ascending=False))


def build_maestro_catalog(maestro_dir: Path, min_performances: int = 50) -> pd.DataFrame:
    """Single-composer MAESTRO performances of composers with enough performances."""
    meta = pd.read_csv(Path(maestro_dir) / "maestro-v3.0.0.csv")
    meta = meta[~meta.canonical_composer.str.contains("/")]
    counts = meta.canonical_composer.value_counts()
    meta = meta[meta.canonical_composer.isin(counts.index[counts >= min_performances])]
    base = Path(maestro_dir)
    return pd.DataFrame({
        "path": [paths.portable(base / f) for f in meta.midi_filename],
        "composer": meta.canonical_composer.to_numpy(),
        "piece_id": (meta.canonical_composer + ":" + meta.canonical_title).to_numpy(),
        "split": meta.split.map(MAESTRO_SPLITS).to_numpy(),
    })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-files", type=int, default=50)
    parser.add_argument("--min-pieces", type=int, default=5)
    args = parser.parse_args(argv)
    paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = build_hf_catalog(paths.HF_DIR)
    raw["piece_id"] = merge_prefix_pieces(raw.piece_id)
    deduped = deduplicate(raw)
    kept, summary = select_composers(deduped, args.min_files, args.min_pieces)
    kept.to_csv(paths.DATA_PROCESSED / "hf_catalog.csv", index=False)
    summary.to_csv(paths.DATA_PROCESSED / "hf_composer_summary.csv")
    print(f"HF: {len(raw)} files, {len(deduped)} after dedup, {len(kept)} kept")
    print(summary.head(25).to_string())

    maestro = build_maestro_catalog(paths.MAESTRO_DIR)
    maestro.to_csv(paths.DATA_PROCESSED / "maestro_catalog.csv", index=False)
    print(pd.crosstab(maestro.composer, maestro.split, margins=True).to_string())


if __name__ == "__main__":
    main()
