import json
import re

import numpy as np
import pandas as pd

from composer_clf import run_tabular


def test_environment_records_commit_and_versions():
    env = run_tabular.environment()
    assert re.fullmatch(r"[0-9a-f]{40}", env["git_commit"])
    assert isinstance(env["git_dirty"], bool)
    assert env["packages"]["lightgbm"] == "4.7.0"
    assert env["python"].startswith("3.13")
