"""Download the two datasets and jSymbolic into data/raw and tools."""
import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from composer_clf import paths

HF_TREE_URL = "https://huggingface.co/api/datasets/drengskapur/midi-classical-music/tree/main?recursive=true"
HF_FILE_URL = "https://huggingface.co/datasets/drengskapur/midi-classical-music/resolve/main/{path}"
MAESTRO_ZIP_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_CSV_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
JSYMBOLIC_URL = "https://sourceforge.net/projects/jmir/files/jSymbolic/jSymbolic%202.2/jSymbolic_2_2_user.zip/download"


def next_link(link_header: str | None) -> str | None:
    """URL of the next page from an HTTP Link header, if any."""
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def fetch(url: str, dest: Path, opener=urllib.request.urlopen, sleep=time.sleep, retries: int = 8) -> Path:
    """Download url to dest unless dest already exists, backing off on HTTP 429/503."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries + 1):
        try:
            with opener(url) as response, open(tmp, "wb") as fh:
                shutil.copyfileobj(response, fh)
            break
        except urllib.error.HTTPError as err:
            if err.code not in (429, 503) or attempt == retries:
                raise
            retry_after = err.headers.get("Retry-After") if err.headers else None
            sleep(float(retry_after) if retry_after else float(2 ** attempt))
    tmp.rename(dest)
    return dest


def list_hf_midi_files() -> list[str]:
    url, files = HF_TREE_URL, []
    while url:
        with urllib.request.urlopen(url) as response:
            entries = json.load(response)
            url = next_link(response.headers.get("Link"))
        files += [e["path"] for e in entries
                  if e["type"] == "file" and e["path"].lower().endswith((".mid", ".midi"))]
    return files


def download_hf(dest: Path = paths.HF_DIR, workers: int = 4) -> int:
    files = list_hf_midi_files()

    def one(rel: str) -> Path:
        return fetch(HF_FILE_URL.format(path=urllib.parse.quote(rel)), dest / Path(rel).name)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files))
    return len(files)


def download_maestro() -> None:
    archive = fetch(MAESTRO_ZIP_URL, paths.DATA_RAW / "maestro-v3.0.0-midi.zip")
    if not paths.MAESTRO_DIR.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(paths.DATA_RAW)
    fetch(MAESTRO_CSV_URL, paths.MAESTRO_DIR / "maestro-v3.0.0.csv")


def download_jsymbolic() -> None:
    archive = fetch(JSYMBOLIC_URL, paths.TOOLS / "jSymbolic_2_2_user.zip")
    if not paths.JSYMBOLIC_JAR.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(paths.TOOLS)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["hf", "maestro", "jsymbolic", "all"])
    args = parser.parse_args(argv)
    if args.what in ("hf", "all"):
        print(f"midi-classical-music: {download_hf()} files")
    if args.what in ("maestro", "all"):
        download_maestro()
        print("MAESTRO done")
    if args.what in ("jsymbolic", "all"):
        download_jsymbolic()
        print(f"jSymbolic at {paths.JSYMBOLIC_JAR}")


if __name__ == "__main__":
    main()
