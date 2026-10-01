"""music21 feature extraction, used only if jSymbolic cannot be run."""
from pathlib import Path

from music21 import converter, features

_LABEL = "label"


def music21_features(path: Path) -> dict[str, float]:
    ds = features.DataSet(classLabel=_LABEL)
    ds.addFeatureExtractors(features.extractorsById("all"))
    ds.addData(converter.parse(str(path)), classValue="x", id=str(path))
    ds.process()
    names, row = ds.getAttributeLabels(), ds.getFeaturesAsList()[0]
    return {n: v for n, v in zip(names, row) if n not in ("Identifier", _LABEL)}
