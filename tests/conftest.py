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
