"""Download the two datasets and jSymbolic into data/raw and tools."""
import json
import re
import urllib.request

HF_TREE_URL = "https://huggingface.co/api/datasets/drengskapur/midi-classical-music/tree/main?recursive=true"


def next_link(link_header: str | None) -> str | None:
    """URL of the next page from an HTTP Link header, if any."""
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def list_hf_midi_files() -> list[str]:
    url, files = HF_TREE_URL, []
    while url:
        with urllib.request.urlopen(url) as response:
            entries = json.load(response)
            url = next_link(response.headers.get("Link"))
        files += [e["path"] for e in entries
                  if e["type"] == "file" and e["path"].lower().endswith((".mid", ".midi"))]
    return files
