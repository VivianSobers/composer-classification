"""Tune, train and test tabular models; every run is written to results/."""
import platform
import subprocess
from importlib import metadata

from composer_clf import paths

PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "music21"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()


def environment(packages=PACKAGES) -> dict:
    """Code version and library versions, stored with every run."""
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(),
            "packages": {p: metadata.version(p) for p in packages}}
