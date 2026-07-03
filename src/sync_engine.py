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
    signals = {cam: _load(path, sr) for cam, path in audio_paths.items()}

    # Shared downsample factor (based on the longest signal) so all cams'
    # lags stay in the same time base.
    max_samples = _MAX_SECONDS * sr
    longest = max(len(y) for y in signals.values())
    factor = int(np.ceil(longest / max_samples)) if longest > max_samples else 1
    work_sr = sr // factor

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
