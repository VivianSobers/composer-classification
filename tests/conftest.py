import numpy as np
import pandas as pd
import pretty_midi
import pytest


def write_midi(path, notes, program=0):
    """notes: list of (pitch, start_s, end_s, velocity)."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for pitch, start, end, velocity in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


SCALE = [(60 + i, i * 0.5, i * 0.5 + 0.5, 80) for i in range(16)]


@pytest.fixture
def midi_factory(tmp_path):
    def make(name, notes=SCALE, program=0):
        return write_midi(tmp_path / name, notes, program)
    return make


@pytest.fixture
def synthetic_tabular():
    """Three composers x 10 pieces x 3 files, with feature F_0 separating the composers."""
    from composer_clf.tabular import TabularData

    rng = np.random.default_rng(0)
    rows = []
    for c, name in enumerate(["aa", "bb", "cc"]):
        for g in range(10):
            for k in range(3):
                rows.append({"path": f"{name}-{g}-{k}.mid", "composer": name, "piece_id": f"{name}:{g}",
                             **{f"F_{j}": rng.normal() + 3 * c * (j == 0) for j in range(6)}})
    frame = pd.DataFrame(rows)
    cols = [f"F_{j}" for j in range(6)]
    return TabularData(X=frame[cols].to_numpy(float), y=frame.composer.to_numpy(),
                       groups=frame.piece_id.to_numpy(), frame=frame, feature_columns=cols)
