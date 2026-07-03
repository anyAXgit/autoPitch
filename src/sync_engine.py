import numpy as np
import librosa
from scipy.signal import correlate

# Signals longer than this are downsampled by an integer factor before
# correlation to bound compute cost.
_MAX_SECONDS = 600


def _load(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def _normalize(x):
    x = x - np.mean(x)
    std = np.std(x)
    return x / std if std > 0 else x


def compute_offsets(audio_paths, ref_cam, sr=16000):
    """Compute per-camera audio offsets (in seconds) relative to ref_cam.

    If the longest signal exceeds `_MAX_SECONDS`, all signals are decimated
    by a shared integer `factor` (`y[::factor]`) before cross-correlation to
    bound compute cost on long footage. This is a plain stride-based
    decimation with NO anti-aliasing filter -- that's fine here because we
    only care about the coarse timing of burst envelopes, not signal
    fidelity. The true sample spacing after `y[::factor]` is the rational
    rate `sr / factor`, not the floored integer `sr // factor`; using the
    floored rate to convert correlation lag back to seconds introduces a
    systematic bias that grows with the offset magnitude, so `work_sr` must
    be computed with float division.
    """
    signals = {cam: _load(path, sr) for cam, path in audio_paths.items()}

    # Shared downsample factor (based on the longest signal) so all cams'
    # lags stay in the same time base.
    max_samples = _MAX_SECONDS * sr
    longest = max(len(y) for y in signals.values())
    factor = int(np.ceil(longest / max_samples)) if longest > max_samples else 1
    # Float division: y[::factor] spaces samples at the true rational rate
    # sr / factor, not the floored integer sr // factor. Using the floored
    # rate here would systematically bias the lag->seconds conversion.
    work_sr = sr / factor

    ref = _normalize(signals[ref_cam][::factor])
    offsets = {ref_cam: 0.0}
    for cam, y in signals.items():
        if cam == ref_cam:
            continue
        sig = _normalize(y[::factor])
        corr = correlate(sig, ref, mode="full")
        lag = np.argmax(corr) - (len(ref) - 1)
        offsets[cam] = lag / work_sr
    return offsets
