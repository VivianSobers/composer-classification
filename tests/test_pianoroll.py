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


def test_shift_positive_and_negative():
    from composer_clf.pianoroll import shift

    roll = (np.arange(64 * 1536).reshape(64, 1536) % 200).astype(np.uint8)
    out = shift(roll, dx=5, dy=2)
    assert (out[2:, 5:] == roll[:-2, :-5]).all()
    assert out[:2].sum() == 0 and out[:, :5].sum() == 0
    back = shift(roll, dx=-5, dy=-2)
    assert (back[:-2, :-5] == roll[2:, 5:]).all()
    assert back[-2:].sum() == 0 and back[:, -5:].sum() == 0


def test_blank_width():
    from composer_clf.pianoroll import blank

    roll = np.ones((64, 1536), np.uint8)
    out = blank(roll, 760)
    assert out[:, 711:810].sum() == 0
    assert out[:, 710].all() and out[:, 810].all()
    assert roll.all()  # input untouched


def test_augment_deterministic_shape():
    from composer_clf.pianoroll import augment

    roll = np.ones((64, 1536), np.uint8)
    a = augment(roll, np.random.default_rng(1))
    b = augment(roll, np.random.default_rng(1))
    assert a.shape == roll.shape and a.dtype == np.uint8 and (a == b).all()
    assert a.sum() < roll.sum()  # the blanked strip always removes something
