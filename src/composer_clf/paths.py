"""Locations of data, tools and outputs, all relative to the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
TOOLS = ROOT / "tools"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

HF_DIR = DATA_RAW / "midi-classical-music"
MAESTRO_DIR = DATA_RAW / "maestro-v3.0.0"
JSYMBOLIC_DIR = TOOLS / "jSymbolic_2_2_user"
JSYMBOLIC_JAR = JSYMBOLIC_DIR / "jSymbolic2.jar"
JSYMBOLIC_CONFIG = JSYMBOLIC_DIR / "jSymbolicDefaultConfigs.txt"


def resolve(rel: str | Path) -> Path:
    """Absolute path for a repo-relative path."""
    return ROOT / rel


def relative(p: str | Path) -> str:
    """Repo-relative POSIX path for a file inside the repo."""
    return Path(p).resolve().relative_to(ROOT).as_posix()
