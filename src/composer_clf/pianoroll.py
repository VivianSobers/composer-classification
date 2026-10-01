"""Piano-roll images as in the paper: 64 pitches x 1536 frames at 10 Hz (153.6 s)."""
import numpy as np
import pretty_midi

FS = 10
N_FRAMES = 1536
LOW, HIGH = 32, 96


def midi_to_roll(path) -> np.ndarray:
    """Velocity roll for MIDI pitches 32-95 over the first 153.6 s, zero-padded."""
    roll = pretty_midi.PrettyMIDI(str(path)).get_piano_roll(fs=FS)
    roll = np.clip(roll[LOW:HIGH, :N_FRAMES], 0, 127)
    out = np.zeros((HIGH - LOW, N_FRAMES), np.uint8)
    out[: roll.shape[0], : roll.shape[1]] = roll
    return out


def shift(roll: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate by dx frames and dy pitches, filling the uncovered edge with zeros."""
    out = np.zeros_like(roll)
    h, w = roll.shape
    src_r, dst_r = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    src_c, dst_c = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[dst_r, dst_c] = roll[src_r, src_c]
    return out


def blank(roll: np.ndarray, z: int) -> np.ndarray:
    """Zero the 99 frames with |j - z| < 50."""
    out = roll.copy()
    out[:, z - 49 : z + 50] = 0
    return out


def augment(roll: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The paper's augmentation: random shift, then a blanked strip near the middle."""
    dx = int(rng.integers(-150, 151))
    dy = int(rng.integers(-10, 11))
    z = int(rng.integers(718, 819))
    return blank(shift(roll, dx, dy), z)
