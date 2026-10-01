import numpy as np

from composer_clf.pianoroll import midi_to_roll


def test_roll_shape_values_and_padding(midi_factory):
    roll = midi_to_roll(midi_factory("n.mid", [(60, 0.0, 1.0, 100)]))
    assert roll.shape == (64, 1536) and roll.dtype == np.uint8
    assert (roll[60 - 32, :10] == 100).all()
    assert roll.sum() == 100 * 10


def test_out_of_range_pitches_dropped(midi_factory):
    roll = midi_to_roll(midi_factory("o.mid", [(20, 0.0, 1.0, 90), (100, 0.0, 1.0, 90)]))
    assert roll.sum() == 0


def test_long_piece_cropped(midi_factory):
    roll = midi_to_roll(midi_factory("l.mid", [(60, 0.0, 300.0, 50)]))
    assert roll.shape == (64, 1536) and (roll[28] == 50).all()


def test_empty_midi(midi_factory):
    assert midi_to_roll(midi_factory("e.mid", [])).sum() == 0
