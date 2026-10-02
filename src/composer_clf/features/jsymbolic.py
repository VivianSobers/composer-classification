"""Run jSymbolic 2.2 over many MIDI files in parallel shards."""
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from composer_clf import paths


def read_jsymbolic_csv(csv_path: Path) -> pd.DataFrame:
    """jSymbolic CSV output with the input file's numeric stem as `index`."""
    df = pd.read_csv(csv_path)
    source = df.columns[0]
    index = df[source].map(lambda s: int(Path(s).stem)).rename("index")
    return pd.concat([df.drop(columns=source), index], axis=1)


def _run_shard(shard_dir: Path, xmx: str, timeout_s: int, jar: Path, config: Path) -> Path:
    values = shard_dir.with_name(shard_dir.name + "_values.xml")
    defs = shard_dir.with_name(shard_dir.name + "_defs.xml")
    cmd = ["java", f"-Xmx{xmx}", "-jar", str(jar), "-configrun", str(config),
           str(shard_dir), str(values), str(defs)]
    with open(shard_dir.with_name(shard_dir.name + ".log"), "w") as log:
        try:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout_s} s\n")
    return values.with_suffix(".csv")


def _run_batch(jobs: dict[int, Path], work_dir: Path, n_shards: int, workers: int,
               xmx: str, timeout_s: int, jar: Path, config: Path) -> pd.DataFrame:
    shard_dirs = [work_dir / f"shard{i:03d}" for i in range(min(n_shards, len(jobs)))]
    for d in shard_dirs:
        d.mkdir(parents=True)
    for k, (index, src) in enumerate(jobs.items()):
        shutil.copy(src, shard_dirs[k % len(shard_dirs)] / f"{index:05d}{src.suffix.lower()}")
    with ThreadPoolExecutor(workers) as pool:
        csvs = list(pool.map(lambda d: _run_shard(d, xmx, timeout_s, jar, config), shard_dirs))
    frames = [read_jsymbolic_csv(c) for c in csvs if c.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["index"])


def _collect(work_dir: Path) -> pd.DataFrame:
    """Every row any earlier run in work_dir managed to extract."""
    frames = [read_jsymbolic_csv(c) for c in sorted(work_dir.rglob("*_values.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["index"])


def extract_jsymbolic(midi_paths, work_dir, n_shards=48, workers=4, xmx="1g", retry_xmx="3g",
                      timeout_s=3600, jar=paths.JSYMBOLIC_JAR, config=paths.JSYMBOLIC_CONFIG, resume=False,
                      retry_workers=2):
    """Returns (features with a leading `path` column in input order, failed paths).

    With resume=True, rows already extracted into work_dir by an earlier (possibly interrupted)
    call with the same file list are reused and only the missing files are run.
    """
    midi_paths = [Path(p) for p in midi_paths]
    work_dir = Path(work_dir)
    if work_dir.exists() and not resume:
        shutil.rmtree(work_dir)
    # Rows are matched to files by position, so a resume is only valid for the identical file list.
    manifest = work_dir / "manifest.txt"
    listing = "".join(f"{p}\n" for p in midi_paths)
    if manifest.exists() and manifest.read_text() != listing:
        raise ValueError(f"{work_dir} was extracted for a different file list; rerun without resume")
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(listing)
    found = _collect(work_dir)
    done = set(found["index"])
    jobs = {i: p for i, p in enumerate(midi_paths) if i not in done}
    if jobs:
        run_dir = work_dir / f"run{len(list(work_dir.glob('run*'))) + 1}"
        new = _run_batch(jobs, run_dir / "pass1", n_shards, workers, xmx, timeout_s, jar, config)
        # A crashed or timed-out JVM loses its whole shard; retry those files one per JVM with more memory.
        new_done = set(new["index"])
        missing = {i: p for i, p in jobs.items() if i not in new_done}
        if missing:
            retry = _run_batch(missing, run_dir / "retry", len(missing), retry_workers,
                               retry_xmx, timeout_s, jar, config)
            new = pd.concat([new, retry], ignore_index=True)
        found = pd.concat([found, new], ignore_index=True) if len(found) else new

    found = found.drop_duplicates("index").sort_values("index").reset_index(drop=True)
    done = set(found["index"])
    failed = [str(p) for i, p in enumerate(midi_paths) if i not in done]
    if found.empty:
        return pd.DataFrame(columns=["path"]), failed
    path = pd.Series([str(midi_paths[i]) for i in found["index"]], name="path")
    return pd.concat([path, found.drop(columns="index")], axis=1), failed
